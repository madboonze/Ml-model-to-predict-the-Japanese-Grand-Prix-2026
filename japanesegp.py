import warnings
warnings.filterwarnings("ignore")

import os
import time
# Matplotlib: use project dir so it works without writable home
_script_dir = os.path.dirname(os.path.abspath(__file__))
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(_script_dir, ".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import fastf1
from fastf1.ergast import Ergast
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Prefer LightGBM, then XGBoost, then sklearn (sklearn works without libomp on Mac)
_lgb, _xgb = None, None
_USE_LIGHTGBM = False
_USE_XGBOOST = False
try:
    import lightgbm as _lgb
    _USE_LIGHTGBM = True
except Exception:
    try:
        import xgboost as _xgb
        _USE_XGBOOST = True
    except Exception:
        pass
if not _USE_LIGHTGBM and not _USE_XGBOOST:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import GradientBoostingClassifier

# Enable FastF1 cache to stay within rate limits and speed up runs
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# --- Constants ---
GP_NAME = "Japan"  # Suzuka
YEARS = [2022, 2023, 2024, 2025, 2026]
SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "R"]
DNF_POSITION = 999
TRAIN_YEARS = [2022, 2023, 2024]   # temporal train
TEST_YEAR = 2025                    # test set
PREDICT_YEAR = 2026                 # final prediction
RACES_BEFORE_JAPAN_FOR_FORM = 3
RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. DATA FETCHING – Suzuka sessions 2022–2026
# =============================================================================

def load_session_safe(year: int, session_type: str):
    """Load a single session; return None if not available or error."""
    try:
        session = fastf1.get_session(year, GP_NAME, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"  [Skip] {year} {session_type}: {e}")
        return None


def fetch_all_suzuka_sessions():
    """
    Fetch FP1/FP2/FP3/Q/R for Japanese GP 2022–2026.
    Special weight in features: 2024–2025 (current regs & form).
    """
    data = {}
    for year in YEARS:
        if year > pd.Timestamp.now().year:
            continue
        data[year] = {}
        for st in SESSION_TYPES:
            sess = load_session_safe(year, st)
            if sess is not None:
                data[year][st] = sess
    return data


def get_constructor_standings_before_round(season: int, round_before: int):
    """
    Get constructor standings after the round preceding the given round.
    Used for team ranking feature (season-to-date before Japan).
    """
    try:
        ergast = Ergast()
        # get standings after round_before (1-indexed in API)
        resp = ergast.get_constructor_standings(season=season, round=round_before)
        if hasattr(resp, "content") and resp.content:
            df = resp.content[0].copy()
            df["season"] = season
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_schedule_and_japan_round(season: int):
    """Get full schedule and Japan round number for the season."""
    try:
        schedule = fastf1.get_event_schedule(season)
        if schedule is None or len(schedule) == 0:
            return None, None
        # Match Japan (Suzuka)
        mask = schedule["EventName"].str.contains("Japan", case=False, na=False)
        if not mask.any():
            mask = schedule["Location"].str.contains("Suzuka", case=False, na=False)
        if mask.any():
            japan_row = schedule[mask].iloc[0]
            return schedule, int(japan_row["RoundNumber"])
        return schedule, None
    except Exception:
        return None, None


# =============================================================================
# 2. FEATURE ENGINEERING – 20+ features per driver-race
# =============================================================================

def _position_from_results(results, use_classified=True):
    """Extract finish position; use 999 for DNF if not classified."""
    pos_col = "ClassifiedPosition" if use_classified else "Position"
    if pos_col not in results.columns:
        return results["Position"] if "Position" in results.columns else None
    pos = results[pos_col].copy()
    # DNF: Status not Finished or ClassifiedPosition NaN
    status = results.get("Status", pd.Series(dtype=object))
    dnf_mask = status.astype(str).str.upper().str.contains("DNF|DNS|DQ|WDL|RET", na=False)
    dnf_mask |= pos.isna()
    pos = pos.astype(float)
    pos.loc[dnf_mask] = DNF_POSITION
    return pos


def _safe_mean(series):
    if series is None or series.empty:
        return np.nan
    return pd.to_numeric(series, errors="coerce").mean()


def _safe_std(series):
    if series is None or series.empty or len(series) < 2:
        return np.nan
    return pd.to_numeric(series, errors="coerce").std()


def _timedelta_to_seconds(series):
    if series is None or series.empty:
        return series
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def build_suzuka_history_features(sessions_dict: dict, year: int, driver_abbr: str):
    """
    Driver performance at Suzuka (2022–present). Weight: 2024–25 40%, 2022–23 20%.
    Features: avg finish, best quali, avg lap time (norm), sector times, podiums/DNFs, position gain/loss.
    """
    feats = {}
    prior_years = [y for y in sessions_dict if y < year and y in YEARS]
    if not prior_years:
        return feats

    finishes_by_year = {}  # year -> finish position (exclude DNF)
    qualis = []
    lap_times_sec = []
    sector1, sector2, sector3 = [], [], []
    podiums = 0
    dnf_count = 0
    grid_positions = []
    position_gains = []

    for y in prior_years:
        year_data = sessions_dict.get(y, {})
        race = year_data.get("R")
        quali = year_data.get("Q")
        if race is not None and hasattr(race, "results") and race.results is not None:
            res = race.results
            driver_res = res[res["Abbreviation"] == driver_abbr]
            if not driver_res.empty:
                pos = _position_from_results(driver_res)
                if pos is not None:
                    p = pos.iloc[0]
                    if p == DNF_POSITION or (np.isfinite(p) and p > 20):
                        dnf_count += 1
                    else:
                        finishes_by_year[y] = float(p)
                        if p <= 3:
                            podiums += 1
                grid = driver_res.get("GridPosition", driver_res.get("Position"))
                if grid is not None and pd.notna(grid.iloc[0]):
                    grid_positions.append(int(grid.iloc[0]))
                    if pos is not None and pos.iloc[0] != DNF_POSITION:
                        position_gains.append(int(grid.iloc[0]) - float(pos.iloc[0]))

        if quali is not None and hasattr(quali, "results") and quali.results is not None:
            qres = quali.results[quali.results["Abbreviation"] == driver_abbr]
            if not qres.empty and "Position" in qres.columns:
                qualis.append(int(qres["Position"].iloc[0]))

        if race is not None and hasattr(race, "laps") and race.laps is not None:
            try:
                driver_laps = race.laps.pick_driver(driver_abbr)
            except Exception:
                driver_laps = race.laps[race.laps["Driver"] == driver_abbr]
            if driver_laps is not None and not driver_laps.empty and "LapTime" in driver_laps.columns:
                lt = _timedelta_to_seconds(driver_laps["LapTime"].dropna())
                if lt is not None and len(lt) > 0:
                    lap_times_sec.extend(lt.dropna().tolist())
                for sc, col in [("Sector1", "Sector1Time"), ("Sector2", "Sector2Time"), ("Sector3", "Sector3Time")]:
                    if col in driver_laps.columns:
                        s = _timedelta_to_seconds(driver_laps[col].dropna())
                        if s is not None and len(s) > 0:
                            (sector1 if sc == "Sector1" else sector2 if sc == "Sector2" else sector3).extend(s.dropna().tolist())

    # Weighted average finish: 2024–25 weight 40% each, 2022–23 weight 20% each (60% total Suzuka)
    if finishes_by_year:
        total_w, w_sum = 0.0, 0.0
        for y, f in finishes_by_year.items():
            w = 0.4 if y >= 2024 else 0.2
            total_w += w * f
            w_sum += w
        feats["suzuka_avg_finish"] = total_w / w_sum if w_sum > 0 else np.mean(list(finishes_by_year.values()))
    else:
        feats["suzuka_avg_finish"] = np.nan

    feats["suzuka_best_quali"] = min(qualis) if qualis else np.nan
    feats["suzuka_podiums"] = podiums
    feats["suzuka_dnf_count"] = dnf_count
    feats["suzuka_avg_position_gain"] = np.mean(position_gains) if position_gains else np.nan

    if lap_times_sec:
        session_avg = np.nanmean(lap_times_sec)
        feats["suzuka_avg_lap_normalized"] = np.nanmean(lap_times_sec) / session_avg if session_avg else np.nan
    else:
        feats["suzuka_avg_lap_normalized"] = np.nan

    for name, sec in [("suzuka_sector1_avg", sector1), ("suzuka_sector2_avg", sector2), ("suzuka_sector3_avg", sector3)]:
        feats[name] = np.nanmean(sec) if sec else np.nan

    return feats


def build_current_form_features(season: int, driver_abbr: str, japan_round: int, ergast: Ergast):
    """
    Current form: last 3 races before Japan. Weight 25%.
    Features: avg finish, avg quali, points, DNF rate, lap time consistency (std of fastest laps).
    """
    feats = {
        "form_avg_finish": np.nan,
        "form_avg_quali": np.nan,
        "form_points": np.nan,
        "form_dnf_rate": np.nan,
        "form_lap_consistency": np.nan,
    }
    if japan_round is None or japan_round <= 1:
        return feats
    rounds_to_use = list(range(max(1, japan_round - RACES_BEFORE_JAPAN_FOR_FORM), japan_round))
    if not rounds_to_use:
        return feats

    finishes, qualis, points_list, dnf_count = [], [], [], 0
    try:
        for r in rounds_to_use:
            results = ergast.get_race_results(season=season, round=r)
            if results is None:
                continue
            res_df = results.content[0] if hasattr(results, "content") and results.content else results
            if not isinstance(res_df, pd.DataFrame) or res_df.empty:
                continue
            # Ergast/Jolpica: driverCode or code for driver abbr
            for col in ["driverCode", "code", "Abbreviation"]:
                if col in res_df.columns:
                    driver_rows = res_df[res_df[col].astype(str).str.upper() == driver_abbr.upper()]
                    break
            else:
                continue
            if driver_rows.empty:
                continue
            row = driver_rows.iloc[0]
            pos_text = str(row.get("positionText", row.get("position", "")))
            if pos_text.upper() in ("R", "DNF", "DNS", "DQ", "WDL", "EX", "NC"):
                dnf_count += 1
            else:
                try:
                    finishes.append(int(float(pos_text)))
                except (ValueError, TypeError):
                    pass
            try:
                points_list.append(float(row.get("points", 0)))
            except (ValueError, TypeError):
                points_list.append(0)
        for r in rounds_to_use:
            q_resp = ergast.get_qualifying_results(season=season, round=r)
            if q_resp is None:
                continue
            q_df = q_resp.content[0] if hasattr(q_resp, "content") and q_resp.content else q_resp
            if not isinstance(q_df, pd.DataFrame) or q_df.empty:
                continue
            for col in ["driverCode", "code", "Abbreviation"]:
                if col in q_df.columns:
                    driver_q = q_df[q_df[col].astype(str).str.upper() == driver_abbr.upper()]
                    if not driver_q.empty and "position" in q_df.columns:
                        qualis.append(int(driver_q["position"].iloc[0]))
                    break
    except Exception:
        pass

    if finishes:
        feats["form_avg_finish"] = np.mean(finishes)
    if qualis:
        feats["form_avg_quali"] = np.mean(qualis)
    if points_list:
        feats["form_points"] = np.sum(points_list)
    n_races = len(rounds_to_use)
    feats["form_dnf_rate"] = dnf_count / n_races if n_races else np.nan
    feats["form_lap_consistency"] = np.std(qualis) if len(qualis) >= 2 else np.nan
    return feats


def build_track_session_features(sessions_dict: dict, year: int, driver_abbr: str):
    """
    Track demands & session data. Weight 10%.
    FP1/FP2/FP3 avg lap delta vs field, qualifying position (Q3 normalized), top speed sectors.
    """
    feats = {
        "fp1_lap_delta": np.nan,
        "fp2_lap_delta": np.nan,
        "fp3_lap_delta": np.nan,
        "quali_position": np.nan,
        "quali_q3_normalized": np.nan,
        "top_speed_sector": np.nan,
    }
    year_data = sessions_dict.get(year, {})
    field_avg = {}

    for st in ["FP1", "FP2", "FP3"]:
        sess = year_data.get(st)
        if sess is None or not hasattr(sess, "laps") or sess.laps is None or sess.laps.empty:
            continue
        laps = sess.laps
        if "LapTime" not in laps.columns:
            continue
        lt_sec = _timedelta_to_seconds(laps["LapTime"])
        if lt_sec is None or lt_sec.empty:
            continue
        valid = lt_sec.dropna()
        if len(valid) == 0:
            continue
        field_avg[st] = valid.mean()
        try:
            driver_laps = sess.laps.pick_driver(driver_abbr)
        except Exception:
            driver_laps = laps[laps["Driver"] == driver_abbr]
        if driver_laps is not None and not driver_laps.empty and "LapTime" in driver_laps.columns:
            d_sec = _timedelta_to_seconds(driver_laps["LapTime"]).dropna()
            if len(d_sec) > 0:
                feats[f"{st.lower()}_lap_delta"] = d_sec.mean() - field_avg[st]

    q = year_data.get("Q")
    if q is not None and hasattr(q, "results") and q.results is not None:
        res = q.results[q.results["Abbreviation"] == driver_abbr]
        if not res.empty:
            feats["quali_position"] = res["Position"].iloc[0]
            if "Q3" in res.columns and pd.notna(res["Q3"].iloc[0]):
                q3 = res["Q3"].iloc[0]
                q3_sec = pd.to_timedelta(q3).total_seconds()
                pole_sec = None
                if "Q3" in q.results.columns:
                    pole = q.results["Q3"].dropna()
                    if len(pole) > 0:
                        pole_sec = pd.to_timedelta(pole.min()).total_seconds()
                if pole_sec is not None and pole_sec > 0:
                    feats["quali_q3_normalized"] = q3_sec / pole_sec

    # Top speed in 130R/Spoon/Esses – use SpeedST or sector speed if available
    for st in ["FP1", "FP2", "FP3", "Q", "R"]:
        sess = year_data.get(st)
        if sess is None or not hasattr(sess, "laps"):
            continue
        try:
            driver_laps = sess.laps.pick_driver(driver_abbr)
        except Exception:
            driver_laps = sess.laps[sess.laps["Driver"] == driver_abbr]
        if driver_laps is not None and not driver_laps.empty:
            for col in ["SpeedST", "SpeedFL", "SpeedI1", "SpeedI2"]:
                if col in driver_laps.columns:
                    s = pd.to_numeric(driver_laps[col], errors="coerce").dropna()
                    if len(s) > 0 and np.isnan(feats["top_speed_sector"]):
                        feats["top_speed_sector"] = s.max()
                    elif len(s) > 0:
                        feats["top_speed_sector"] = max(feats["top_speed_sector"], s.max())
                    break
            break
    return feats


def build_external_features(season: int, round_num: int, driver_abbr: str,
                            sessions_dict: dict, constructor_standings: pd.DataFrame):
    """
    External factors. Weight 5%.
    Weather (temp, rain prob, track temp), starting position, constructor rank, safety car likelihood.
    """
    feats = {
        "air_temperature": np.nan,
        "track_temperature": np.nan,
        "starting_position": np.nan,
        "constructor_rank": np.nan,
        "safety_car_likelihood": 0.35,  # historical Suzuka average placeholder
    }
    year_data = sessions_dict.get(season, {})
    race = year_data.get("R")
    if race is None:
        quali = year_data.get("Q")
        if quali is not None and hasattr(quali, "results"):
            res = quali.results[quali.results["Abbreviation"] == driver_abbr]
            if not res.empty:
                feats["starting_position"] = res["Position"].iloc[0]
    else:
        if hasattr(race, "weather_data") and race.weather_data is not None and not race.weather_data.empty:
            w = race.weather_data
            if "AirTemp" in w.columns:
                feats["air_temperature"] = w["AirTemp"].mean()
            if "TrackTemp" in w.columns:
                feats["track_temperature"] = w["TrackTemp"].mean()
        if hasattr(race, "results") and race.results is not None:
            res = race.results[race.results["Abbreviation"] == driver_abbr]
            if not res.empty:
                feats["starting_position"] = res.get("GridPosition", res["Position"]).iloc[0]
    if constructor_standings is not None and not constructor_standings.empty and "constructorName" in constructor_standings.columns:
        team_name = None
        for sess_key in ["R", "Q", "FP1", "FP2", "FP3"]:
            sess = year_data.get(sess_key)
            if sess is not None and hasattr(sess, "results") and sess.results is not None:
                res = sess.results[sess.results["Abbreviation"] == driver_abbr]
                if not res.empty and "TeamName" in res.columns:
                    team_name = res["TeamName"].iloc[0]
                    break
        if team_name:
            # Match constructor name (Ergast may use different spelling)
            for _, c in constructor_standings.iterrows():
                if c.get("constructorName") and team_name and str(c["constructorName"]).upper() in str(team_name).upper():
                    feats["constructor_rank"] = c.get("position", np.nan)
                    break
    return feats


def build_target_and_quali(sessions_dict: dict, year: int):
    """Get race result (position, DNF=999) and podium binary per driver for a given year."""
    year_data = sessions_dict.get(year, {})
    race = year_data.get("R")
    if race is None or not hasattr(race, "results") or race.results is None:
        return None, None
    results = race.results.copy()
    pos = _position_from_results(results)
    results["finish_position"] = pos
    results["podium"] = (pos <= 3) & (pos != DNF_POSITION)
    results["driver_abbr"] = results["Abbreviation"]
    return results[["driver_abbr", "finish_position", "podium"]], results


def build_full_dataset(sessions_dict: dict):
    """
    Build driver-race rows with 20+ features and targets.
    Rows: (driver_abbr, year) for each year where we have race or qualifying data.
    """
    ergast = Ergast()
    rows = []
    all_drivers_per_year = {}

    for year in YEARS:
        if year > pd.Timestamp.now().year:
            continue
        year_data = sessions_dict.get(year, {})
        # Determine drivers from race or qualifying
        drivers = set()
        for st in ["R", "Q", "FP1", "FP2", "FP3"]:
            sess = year_data.get(st)
            if sess is not None and hasattr(sess, "results") and sess.results is not None:
                drivers.update(sess.results["Abbreviation"].astype(str).tolist())
        if not drivers:
            continue
        all_drivers_per_year[year] = list(drivers)

    for year in list(all_drivers_per_year.keys()):
        schedule, japan_round = get_schedule_and_japan_round(year)
        constructor_standings = pd.DataFrame()
        # Standings before Japan = after previous round
        if japan_round is not None and japan_round > 1:
            constructor_standings = get_constructor_standings_before_round(year, japan_round - 1)
        if constructor_standings is None:
            constructor_standings = pd.DataFrame()

        for driver_abbr in all_drivers_per_year[year]:
            row = {"year": year, "driver_abbr": driver_abbr}
            row.update(build_suzuka_history_features(sessions_dict, year, driver_abbr))
            row.update(build_current_form_features(year, driver_abbr, japan_round, ergast))
            row.update(build_track_session_features(sessions_dict, year, driver_abbr))
            row.update(build_external_features(year, japan_round or 0, driver_abbr, sessions_dict, constructor_standings))

            target_df, _ = build_target_and_quali(sessions_dict, year)
            if target_df is not None:
                tr = target_df[target_df["driver_abbr"] == driver_abbr]
                if not tr.empty:
                    row["finish_position"] = tr["finish_position"].iloc[0]
                    row["podium"] = int(tr["podium"].iloc[0])
                else:
                    row["finish_position"] = np.nan
                    row["podium"] = np.nan
            else:
                row["finish_position"] = np.nan
                row["podium"] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# 3. TRAIN/TEST SPLIT & MODEL
# =============================================================================

def temporal_split(df: pd.DataFrame):
    """
    80/20 temporal split:
    Train: 2022, 2023, 2024 (+ 2025 FP/Q only for feature availability; no 2025 race in train).
    Test: 2025 Race.
    """
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    test = df[df["year"] == TEST_YEAR].copy()
    return train, test


def get_feature_columns():
    """All feature column names (exclude target and identifiers)."""
    return [
        "suzuka_avg_finish", "suzuka_best_quali", "suzuka_avg_lap_normalized",
        "suzuka_sector1_avg", "suzuka_sector2_avg", "suzuka_sector3_avg",
        "suzuka_podiums", "suzuka_dnf_count", "suzuka_avg_position_gain",
        "form_avg_finish", "form_avg_quali", "form_points", "form_dnf_rate", "form_lap_consistency",
        "fp1_lap_delta", "fp2_lap_delta", "fp3_lap_delta",
        "quali_position", "quali_q3_normalized", "top_speed_sector",
        "air_temperature", "track_temperature", "starting_position", "constructor_rank", "safety_car_likelihood",
    ]


def prepare_xy(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    """Prepare X_train, y_train, X_test, y_test; fill NaN with median."""
    X_train = train[feature_cols].copy()
    y_train = train["finish_position"].copy()
    X_test = test[feature_cols].copy()
    y_test = test["finish_position"].copy()

    # Drop rows where target is NaN (e.g. no race yet)
    train_valid = y_train.notna()
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]
    test_valid = y_test.notna()
    X_test = X_test[test_valid]
    y_test = y_test[test_valid]

    # Fill NaN features with median of training set
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    return X_train, y_train, X_test, y_test, train[train_valid], test[test_valid]


def accuracy_at_k(y_true, y_pred, k=3):
    """Accuracy@k: proportion of samples where true label is in top-k predictions (by predicted position)."""
    if len(y_true) == 0:
        return 0.0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # For each sample, check if true position is in top k (1..k)
    correct = (y_true <= k) & (np.round(y_pred) <= k)
    return correct.mean()


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols):
    """Train LightGBM or XGBoost with CV, tune key hyperparams (runtime < 10 min)."""
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    best_mae = np.inf
    best_params = None
    best_model = None
    tscv = TimeSeriesSplit(n_splits=3)
    keys = list(param_grid.keys())
    n_combinations = np.prod([len(param_grid[k]) for k in keys])
    from itertools import product
    count = 0
    for values in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))
        if _USE_LIGHTGBM:
            model = _lgb.LGBMRegressor(
                **params,
                random_state=RANDOM_STATE,
                verbosity=-1,
                force_col_wise=True,
            )
        elif _USE_XGBOOST:
            model = _xgb.XGBRegressor(
                **params,
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=1,
            )
        else:
            # sklearn GradientBoostingRegressor: colsample_bytree -> max_features
            p = {k: v for k, v in params.items() if k != "colsample_bytree"}
            p["max_features"] = params.get("colsample_bytree", 1.0)
            model = GradientBoostingRegressor(**p, random_state=RANDOM_STATE)
        mae_scores = -cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1)
        mae = mae_scores.mean()
        count += 1
        if count <= 3 or count % 10 == 0 or count == n_combinations:
            print(f"  CV try {count}/{n_combinations}: MAE={mae:.3f}")
        if mae < best_mae:
            best_mae = mae
            best_params = params
            best_model = model
    print(f"  Best CV MAE: {best_mae:.3f} with {best_params}")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    acc3 = accuracy_at_k(y_test, y_pred, k=3)
    print(f"  Test MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}, Accuracy@3: {acc3:.3f}")
    return best_model, best_params, {"mae": mae, "rmse": rmse, "r2": r2, "accuracy_at_3": acc3}, y_pred


# =============================================================================
# 4. FEATURE IMPORTANCE & 2026 PREDICTIONS
# =============================================================================

def plot_feature_importance(model, feature_cols, path: str):
    """Plot and save feature importance (gain). Suzuka/form features expected in top 3."""
    try:
        sns.set_theme(style="whitegrid")
    except Exception:
        pass
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(feature_cols))
    colors = ["#2ecc71" if ("suzuka" in feature_cols[i] or "form" in feature_cols[i]) else "steelblue" for i in idx]
    ax.barh(y_pos, imp[idx], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_cols[i] for i in idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance – Suzuka GP Predictor")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance saved: {path}")


def predict_2026_podium_probabilities(model, df_full: pd.DataFrame, feature_cols: str, sessions_dict: dict):
    """
    Predict 2026 Japanese GP: finish position and podium probability.
    Uses 2026 data if available; else uses 2025 form + Suzuka history.
    """
    pred_2026 = df_full[df_full["year"] == PREDICT_YEAR].copy()
    if pred_2026.empty:
        # Fallback: use 2025 drivers and last known features
        pred_2026 = df_full[df_full["year"] == TEST_YEAR].copy()
        pred_2026["year"] = PREDICT_YEAR
        print("  [Info] No 2026 data; using 2025 driver list and features as proxy.")
    if pred_2026.empty:
        return pd.DataFrame(), np.array([]), np.array([])

    medians = df_full[df_full["year"].isin(TRAIN_YEARS)][feature_cols].median()
    X_2026 = pred_2026[feature_cols].fillna(medians)
    pos_pred = model.predict(X_2026)

    # Podium probability: use classifier or derive from position distribution
    # We train a simple podium classifier on same features for probability
    train = df_full[df_full["year"].isin(TRAIN_YEARS)].dropna(subset=["podium"])
    if len(train) >= 10:
        if _USE_LIGHTGBM:
            clf = _lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE, verbosity=-1)
        elif _USE_XGBOOST:
            clf = _xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE, verbosity=0)
        else:
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
        X_tr = train[feature_cols].fillna(train[feature_cols].median())
        clf.fit(X_tr, train["podium"].astype(int))
        podium_proba = clf.predict_proba(X_2026)[:, 1]
    else:
        # Heuristic: inverse of predicted position
        podium_proba = 1.0 / (np.clip(pos_pred, 1, 20) + 0.5)
        podium_proba = podium_proba / podium_proba.sum() * min(3, len(podium_proba))

    pred_2026["predicted_position"] = pos_pred
    pred_2026["podium_probability"] = podium_proba
    return pred_2026, pos_pred, podium_proba


def bootstrap_confidence_interval(model, X_2026, n_bootstrap=200, alpha=0.05):
    """Simple bootstrap 95% CI for predicted position (per driver)."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X_2026)
    preds = np.zeros((n_bootstrap, n))
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        preds[b] = model.predict(X_2026.iloc[idx])
    return np.percentile(preds, 100 * alpha / 2, axis=0), np.percentile(preds, 100 * (1 - alpha / 2), axis=0)


def print_output(model, metrics, feature_cols, pred_2026_df):
    """Print all key results to console: metrics, success criteria, feature importance, predictions."""
    print("\n")
    print("=" * 70)
    print("                         OUTPUT – RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Evaluation metrics (test set) ---")
    print(f"  Mean Absolute Error (MAE):     {metrics['mae']:.4f}")
    print(f"  RMSE:                          {metrics['rmse']:.4f}")
    print(f"  R² score:                      {metrics['r2']:.4f}")
    print(f"  Accuracy@3 (top-3 correct):     {metrics['accuracy_at_3']*100:.2f}%")

    print("\n--- Success criteria ---")
    mae_ok = metrics["mae"] < 4.5
    acc3_ok = metrics["accuracy_at_3"] > 0.65
    imp_order = np.argsort(model.feature_importances_)[::-1]
    top3_names = [feature_cols[i] for i in imp_order[:3]]
    suzuka_form_in_top3 = any("suzuka" in f or "form" in f for f in top3_names)
    print(f"  MAE < 4.5:              {metrics['mae']:.3f}  {'PASS' if mae_ok else 'FAIL'}")
    print(f"  Accuracy@3 > 65%:      {metrics['accuracy_at_3']*100:.1f}%  {'PASS' if acc3_ok else 'FAIL'}")
    print(f"  Suzuka/form in top 3:  {top3_names}  {'PASS' if suzuka_form_in_top3 else 'FAIL'}")

    print("\n--- Feature importance (top 10) ---")
    for i, idx in enumerate(imp_order[:10], 1):
        print(f"  {i:2}. {feature_cols[idx]:30s}  {model.feature_importances_[idx]:.0f}")

    if pred_2026_df is not None and not pred_2026_df.empty:
        print("\n--- Top 5 predicted finishers – 2026 Japanese GP (with 95% CI) ---")
        top5 = pred_2026_df.head(5)[["rank", "driver_abbr", "predicted_position", "podium_probability", "position_ci_lower", "position_ci_upper"]]
        print(top5.to_string(index=False))
        print("\n--- Full 2026 predicted finishing order ---")
        full = pred_2026_df[["rank", "driver_abbr", "predicted_position", "podium_probability"]].copy()
        full["predicted_position"] = full["predicted_position"].round(2)
        full["podium_probability"] = full["podium_probability"].round(4)
        print(full.to_string(index=False))
    else:
        print("\n--- 2026 predictions: (no prediction data) ---")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    start = time.time()
    backend = "LightGBM" if _USE_LIGHTGBM else "XGBoost" if _USE_XGBOOST else "sklearn GradientBoosting"
    print("=" * 60)
    print("Suzuka Japanese GP 2026 – Predictive Model Pipeline")
    print("=" * 60)
    print(f"Model backend: {backend}")

    # 1. Fetch data
    print("\n[1/7] Fetching Suzuka sessions 2022–2026...")
    sessions_dict = fetch_all_suzuka_sessions()
    for y, v in sessions_dict.items():
        print(f"  {y}: {list(v.keys())}")

    # 2. Feature engineering
    print("\n[2/7] Building feature matrix (20+ features per driver-race)...")
    df = build_full_dataset(sessions_dict)
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    print(f"  Rows: {len(df)}, Features: {len(feature_cols)}")

    # 3. Temporal split
    print("\n[3/7] 80/20 temporal split (train: 2022–2024, test: 2025)...")
    train_df, test_df = temporal_split(df)
    X_train, y_train, X_test, y_test, train_clean, test_clean = prepare_xy(train_df, test_df, feature_cols)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    if len(X_train) < 5 or len(X_test) < 1:
        print("\n[WARN] Insufficient data after split. Ensure 2022–2025 Japan sessions are available.")
        print("  Attempting to continue with 2026 prediction using 2025 as proxy...")
        if len(X_train) < 2:
            print("  Aborting: need at least 2 training samples.")
            return

    # 4. Train & evaluate
    print("\n[4/7] Training LightGBM (hyperparameter search)...")
    model, best_params, metrics, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols)

    # 5. Success criteria
    print("\n[5/7] Success criteria check:")
    mae_ok = metrics["mae"] < 4.5
    acc3_ok = metrics["accuracy_at_3"] > 0.65
    print(f"  MAE < 4.5: {metrics['mae']:.3f} {'PASS' if mae_ok else 'FAIL'}")
    print(f"  Accuracy@3 > 65%: {metrics['accuracy_at_3']*100:.1f}% {'PASS' if acc3_ok else 'FAIL'}")
    imp_order = np.argsort(model.feature_importances_)[::-1]
    top3_names = [feature_cols[i] for i in imp_order[:3]]
    suzuka_form_in_top3 = any("suzuka" in f or "form" in f for f in top3_names)
    print(f"  Top-3 features: {top3_names}")
    print(f"  Suzuka/form in top 3: {'PASS' if suzuka_form_in_top3 else 'FAIL'}")

    # 6. Feature importance plot
    print("\n[6/7] Feature importance and visualizations...")
    plot_feature_importance(model, feature_cols, os.path.join(OUTPUT_DIR, "feature_importance.png"))

    # 7. 2026 predictions
    print("\n[7/7] Predicting 2026 Japanese GP...")
    pred_2026_df, pos_2026, podium_proba_2026 = predict_2026_podium_probabilities(
        model, df, feature_cols, sessions_dict
    )
    if not pred_2026_df.empty:
        pred_2026_df = pred_2026_df.sort_values("predicted_position").reset_index(drop=True)
        pred_2026_df["rank"] = np.arange(1, len(pred_2026_df) + 1)
        X_2026 = pred_2026_df[feature_cols].fillna(df[df["year"].isin(TRAIN_YEARS)][feature_cols].median())
        ci_lo, ci_hi = bootstrap_confidence_interval(model, X_2026)
        pred_2026_df["position_ci_lower"] = ci_lo
        pred_2026_df["position_ci_upper"] = ci_hi
        top5 = pred_2026_df.head(5)[["rank", "driver_abbr", "predicted_position", "podium_probability", "position_ci_lower", "position_ci_upper"]]
        print("\n  Top 5 predicted finishers (2026 Japanese GP) with 95% CI:")
        print(top5.to_string(index=False))
        pred_2026_df.to_csv(os.path.join(OUTPUT_DIR, "2026_predictions.csv"), index=False)

        # Podium probability bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        drivers = pred_2026_df["driver_abbr"].astype(str)
        probs = pred_2026_df["podium_probability"].values
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(drivers)))
        ax.barh(drivers, probs, color=colors)
        ax.set_xlabel("Podium probability")
        ax.set_title("2026 Japanese GP – Predicted Podium Probability per Driver")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2026_podium_probability.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Podium probability chart saved: {OUTPUT_DIR}/2026_podium_probability.png")
    else:
        print("  No 2026 prediction dataframe produced.")

    # Consolidated output to console
    print_output(model, metrics, feature_cols, pred_2026_df if not pred_2026_df.empty else None)

    elapsed = time.time() - start
    print(f"Total runtime: {elapsed/60:.2f} minutes")
    print("Pipeline complete.")
    return model, df, metrics, pred_2026_df


if __name__ == "__main__":
    main()
