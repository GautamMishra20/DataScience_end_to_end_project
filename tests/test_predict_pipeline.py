import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipelines.predict import CustomData, PredictPipeline

def test_prediction_returns_numeric():
    """Smoke test: pipeline loads artifacts and returns a numeric prediction."""
    data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        reading_score=72.0,
        writing_score=74.0
    )
    df = data.get_data_as_data_frame()
    pipeline = PredictPipeline()
    result = pipeline.predict(df)

    assert result is not None
    assert len(result) == 1
    assert isinstance(float(result[0]), float)
    assert 0 <= float(result[0]) <= 100


def test_prediction_higher_scores_give_higher_result():
    """Sanity check: higher reading/writing scores should predict higher math score."""
    low = CustomData(
        gender="male",
        race_ethnicity="group A",
        parental_level_of_education="some high school",
        lunch="free/reduced",
        test_preparation_course="none",
        reading_score=30.0,
        writing_score=28.0
    )
    high = CustomData(
        gender="male",
        race_ethnicity="group A",
        parental_level_of_education="some high school",
        lunch="free/reduced",
        test_preparation_course="none",
        reading_score=90.0,
        writing_score=92.0
    )
    pipeline = PredictPipeline()
    low_result  = pipeline.predict(low.get_data_as_data_frame())[0]
    high_result = pipeline.predict(high.get_data_as_data_frame())[0]

    assert high_result > low_result