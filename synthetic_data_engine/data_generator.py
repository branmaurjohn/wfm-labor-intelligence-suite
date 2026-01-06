import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from faker import Faker


@dataclass
class Config:
    seed: int
    days: int
    orgs: list
    jobs: dict
    paycodes: list


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        seed=int(raw["seed"]),
        days=int(raw["days"]),
        orgs=list(raw["orgs"]),
        jobs=dict(raw["jobs"]),
        paycodes=list(raw["paycodes"]),
    )


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def shift_bucket_from_start(start_hour: int) -> str:
    if 6 <= start_hour < 14:
        return "DAY"
    if 14 <= start_hour < 22:
        return "EVENING"
    return "NIGHT"


def generate_employees(cfg: Config, fake: Faker, n_per_org: int = 80) -> pd.DataFrame:
    rows = []
    emp_num = 100000
    for org_path in cfg.orgs:
        dept = org_path.split("/")[-1]
        dept_jobs = cfg.jobs.get(dept, [])
        for _ in range(n_per_org):
            job = random.choice(dept_jobs) if dept_jobs else {
                "job_code": "GEN",
                "job_title": "General Staff",
                "job_family": dept
            }
            rows.append({
                "employee_id": f"E{emp_num}",
                "employee_number": emp_num,
                "employee_name": fake.name(),
                "org_path": org_path,
                "department": dept,
                "job_code": job["job_code"],
                "job_title": job["job_title"],
                "job_family": job["job_family"],
                "employment_status": random.choice(["FT", "PT", "PRN"]),
                "home_org_path": org_path,
                "hourly_rate": round(random.uniform(18, 55) if dept == "Nursing" else random.uniform(15, 35), 2),
            })
            emp_num += 1
    return pd.DataFrame(rows)


def generate_schedule(cfg: Config, employees: pd.DataFrame, start_date: date) -> pd.DataFrame:
    
    rows = []
    shift_id = 500000
    for d in range(cfg.days):
        day = start_date + timedelta(days=d)
        dow = day.weekday()  # 0=Mon
        for org_path in cfg.orgs:
            dept = org_path.split("/")[-1]

            base = 55 if dept == "Nursing" else 22 if dept == "EVS" else 10
            if dept == "HR" and dow >= 5:
                base = 3
            demand = max(0, int(np.random.normal(loc=base, scale=3)))

            buckets = [("DAY", 0.45), ("EVENING", 0.35), ("NIGHT", 0.20)]
            for bucket, pct in buckets:
                needed = int(round(demand * pct))
                open_count = int(round(needed * random.uniform(0.03, 0.08)))

                org_emps = employees[employees["org_path"] == org_path]
                if org_emps.empty:
                    continue
                scheduled_count = max(0, needed - open_count)
                chosen = org_emps.sample(n=min(scheduled_count, len(org_emps)), replace=False)

                if bucket == "DAY":
                    start_hour = random.choice([6, 7])
                elif bucket == "EVENING":
                    start_hour = random.choice([14, 15])
                else:
                    start_hour = random.choice([22, 23])

                for _, e in chosen.iterrows():
                    start_dtm = datetime(day.year, day.month, day.day, start_hour, 0, 0)
                    end_dtm = start_dtm + timedelta(hours=12 if dept == "Nursing" else 8)
                    rows.append({
                        "schedule_date": day.isoformat(),
                        "shift_id": f"S{shift_id}",
                        "employee_id": e["employee_id"],
                        "org_path": org_path,
                        "job_code": e["job_code"],
                        "shift_start": start_dtm.isoformat(sep=" "),
                        "shift_end": end_dtm.isoformat(sep=" "),
                        "shift_bucket": bucket,
                        "is_open_shift": False,
                        "scheduled_hours": (end_dtm - start_dtm).total_seconds() / 3600.0,
                    })
                    shift_id += 1

                for _ in range(open_count):
                    start_dtm = datetime(day.year, day.month, day.day, start_hour, 0, 0)
                    end_dtm = start_dtm + timedelta(hours=12 if dept == "Nursing" else 8)
                    rows.append({
                        "schedule_date": day.isoformat(),
                        "shift_id": f"S{shift_id}",
                        "employee_id": None,
                        "org_path": org_path,
                        "job_code": random.choice(cfg.jobs.get(dept, [{"job_code": "GEN"}]))["job_code"],
                        "shift_start": start_dtm.isoformat(sep=" "),
                        "shift_end": end_dtm.isoformat(sep=" "),
                        "shift_bucket": bucket,
                        "is_open_shift": True,
                        "scheduled_hours": (end_dtm - start_dtm).total_seconds() / 3600.0,
                    })
                    shift_id += 1

    return pd.DataFrame(rows)


def generate_timecards(cfg: Config, employees: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:

    rows = []
    tc_id = 900000

    scheduled_assigned = schedule[schedule["is_open_shift"] == False].copy()
    scheduled_assigned["shift_start_dt"] = pd.to_datetime(scheduled_assigned["shift_start"])
    scheduled_assigned["shift_end_dt"] = pd.to_datetime(scheduled_assigned["shift_end"])

    for _, s in scheduled_assigned.iterrows():
        dept = s["org_path"].split("/")[-1]
        start = s["shift_start_dt"].to_pydatetime()
        end = s["shift_end_dt"].to_pydatetime()

        late_minutes = int(np.random.normal(loc=4, scale=6))
        early_out_minutes = int(np.random.normal(loc=3, scale=8))

        clock_in = start + timedelta(minutes=max(-15, late_minutes))
        clock_out = end - timedelta(minutes=max(-30, early_out_minutes))

        worked_hours = max(0.0, (clock_out - clock_in).total_seconds() / 3600.0)

        paycode = "REG"
        if dept == "Nursing" and worked_hours > 12.0:
            paycode = "OT"
        elif dept == "Nursing" and random.random() < 0.18:
            paycode = random.choice(["REG", "OT", "CALL"])
        elif dept != "Nursing" and random.random() < 0.08:
            paycode = "OT"

        rows.append({
            "timecard_entry_id": f"T{tc_id}",
            "employee_id": s["employee_id"],
            "work_date": s["schedule_date"],
            "org_path": s["org_path"],
            "home_org_path": employees.set_index("employee_id").loc[s["employee_id"], "home_org_path"],
            "job_code": s["job_code"],
            "clock_in": clock_in.isoformat(sep=" "),
            "clock_out": clock_out.isoformat(sep=" "),
            "worked_hours": round(worked_hours, 2),
            "paycode": paycode,
            "scheduled_shift_id": s["shift_id"],
        })
        tc_id += 1

    for _ in range(int(len(scheduled_assigned) * 0.02)):
        e = employees.sample(1).iloc[0]
        day = pd.to_datetime(scheduled_assigned.sample(1)["schedule_date"].iloc[0]).date()
        start_hour = random.choice([5, 9, 13, 17, 21])
        start = datetime(day.year, day.month, day.day, start_hour, 0, 0)
        end = start + timedelta(hours=random.choice([4, 6, 8]))
        rows.append({
            "timecard_entry_id": f"T{tc_id}",
            "employee_id": e["employee_id"],
            "work_date": day.isoformat(),
            "org_path": e["org_path"],
            "home_org_path": e["home_org_path"],
            "job_code": e["job_code"],
            "clock_in": start.isoformat(sep=" "),
            "clock_out": end.isoformat(sep=" "),
            "worked_hours": round((end - start).total_seconds() / 3600.0, 2),
            "paycode": random.choice(["REG", "OT", "CALL"]),
            "scheduled_shift_id": None,
        })
        tc_id += 1

    return pd.DataFrame(rows)


def main():
    cfg = load_config("synthetic_data_engine/config.yaml")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    fake = Faker()
    Faker.seed(cfg.seed)

    outdir = Path("data/synthetic_raw")
    ensure_outdir(outdir)

    start_date = (datetime.utcnow().date() - timedelta(days=cfg.days))

    employees = generate_employees(cfg, fake, n_per_org=70)
    schedule = generate_schedule(cfg, employees, start_date=start_date)
    timecards = generate_timecards(cfg, employees, schedule)

    employees.to_csv(outdir / "employees.csv", index=False)
    schedule.to_csv(outdir / "schedules.csv", index=False)
    timecards.to_csv(outdir / "timecards.csv", index=False)

    print("Synthetic data generated:")
    print(f"- {outdir / 'employees.csv'} ({len(employees):,} rows)")
    print(f"- {outdir / 'schedules.csv'} ({len(schedule):,} rows)")
    print(f"- {outdir / 'timecards.csv'} ({len(timecards):,} rows)")


if __name__ == "__main__":
    main()
