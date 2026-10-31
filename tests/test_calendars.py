import pandas as pd

from src.data.calendars import calendar_features, portugal_holidays


def test_calendar_encodings_have_expected_shape_and_future_known_values():
    timestamps = pd.date_range("2014-12-24", periods=48, freq="h")

    cyclic = calendar_features(timestamps, encoding="cyclic")
    onehot = calendar_features(timestamps, encoding="onehot")

    assert cyclic.shape == (48, 7)
    assert onehot.shape == (48, 44)
    assert cyclic.loc[pd.Timestamp("2014-12-25 12:00"), "is_holiday"] == 1
    assert onehot.index.equals(timestamps)


def test_portugal_holiday_names_are_available():
    holidays_by_date = portugal_holidays([2014])

    assert "Natal" in holidays_by_date[pd.Timestamp("2014-12-25").date()]
