import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import SessionLocal
from app.models.audience_segment import AudienceSegment
from app.models.category import Category
from app.models.category_audience_score import CategoryAudienceScore


# Adjust these mappings if the Google Form used a different encoding.
GENDER_MAP = {
    "0": "male",
    "1": "female",
}

AGE_CODE_MAP = {
    "1": {
        "age_group": "18-25",
        "age_min": 18,
        "age_max": 25,
    },
    "2": {
        "age_group": "26-35",
        "age_min": 26,
        "age_max": 35,
    },
    "3": {
        "age_group": "36-45",
        "age_min": 36,
        "age_max": 45,
    },
}

CATEGORY_NAME_MAP = {
    "Entertainment": "Entertainment",
    "Fashion": "Fashion",
    "Food": "Food",
    "Health_and_Beauty": "Health and Beauty",
    "Tech": "Tech",
    "Travel": "Travel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed categories, audience segments, and category scores from a probability matrix CSV.",
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Absolute or relative path to the probability matrix CSV file.",
    )
    parser.add_argument(
        "--keep-current-score",
        action="store_true",
        help="Do not overwrite current_score for existing rows.",
    )
    return parser.parse_args()


def normalize_code(raw_value: object) -> str:
    return str(raw_value).strip().replace(".0", "")


def get_or_create_category(db, name: str) -> Category:
    stmt = select(Category).where(Category.name == name)
    category = db.execute(stmt).scalar_one_or_none()

    if category is None:
        category = Category(name=name)
        db.add(category)
        db.flush()

    return category


def get_or_create_audience_segment(
    db,
    gender: str,
    age_group: str,
    age_min: int,
    age_max: int,
) -> AudienceSegment:
    stmt = select(AudienceSegment).where(
        AudienceSegment.gender == gender,
        AudienceSegment.age_min == age_min,
        AudienceSegment.age_max == age_max,
    )
    audience_segment = db.execute(stmt).scalar_one_or_none()

    if audience_segment is None:
        audience_segment = AudienceSegment(
            gender=gender,
            age_group=age_group,
            age_min=age_min,
            age_max=age_max,
        )
        db.add(audience_segment)
        db.flush()

    return audience_segment


def upsert_category_audience_score(
    db,
    category_id: int,
    audience_segment_id: int,
    probability: float,
    keep_current_score: bool,
) -> CategoryAudienceScore:
    stmt = select(CategoryAudienceScore).where(
        CategoryAudienceScore.category_id == category_id,
        CategoryAudienceScore.audience_segment_id == audience_segment_id,
    )
    score = db.execute(stmt).scalar_one_or_none()

    if score is None:
        score = CategoryAudienceScore(
            category_id=category_id,
            audience_segment_id=audience_segment_id,
            initial_score=probability,
            current_score=probability,
        )
        db.add(score)
        db.flush()
        return score

    score.initial_score = probability
    if not keep_current_score:
        score.current_score = probability

    db.flush()
    return score


def seed_from_probability_matrix(csv_path: Path, keep_current_score: bool) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    category_columns = [
        column_name
        for column_name in reader.fieldnames or []
        if column_name not in {"gender", "age_code"}
    ]

    unknown_categories = [
        category_column
        for category_column in category_columns
        if category_column not in CATEGORY_NAME_MAP
    ]
    if unknown_categories:
        raise ValueError(
            "Missing category mapping for columns: "
            + ", ".join(sorted(unknown_categories)),
        )

    db = SessionLocal()
    try:
        categories_by_column = {
            category_column: get_or_create_category(
                db=db,
                name=CATEGORY_NAME_MAP[category_column],
            )
            for category_column in category_columns
        }

        for row in rows:
            gender_code = normalize_code(row["gender"])
            age_code = normalize_code(row["age_code"])

            if gender_code not in GENDER_MAP:
                raise ValueError(f"Unknown gender code in CSV: {gender_code}")

            if age_code not in AGE_CODE_MAP:
                raise ValueError(f"Unknown age_code in CSV: {age_code}")

            age_info = AGE_CODE_MAP[age_code]
            audience_segment = get_or_create_audience_segment(
                db=db,
                gender=GENDER_MAP[gender_code],
                age_group=age_info["age_group"],
                age_min=age_info["age_min"],
                age_max=age_info["age_max"],
            )

            for category_column in category_columns:
                probability = float(row[category_column])
                category = categories_by_column[category_column]
                upsert_category_audience_score(
                    db=db,
                    category_id=category.id,
                    audience_segment_id=audience_segment.id,
                    probability=probability,
                    keep_current_score=keep_current_score,
                )

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    seed_from_probability_matrix(
        csv_path=csv_path,
        keep_current_score=args.keep_current_score,
    )
    print("Seed probability matrix completed successfully.")


if __name__ == "__main__":
    main()
