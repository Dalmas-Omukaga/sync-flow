import pandas as pd
import time
from pathlib import Path


DATA_PATH = "data/live/focus_predictions_live.csv"


class FocusAnalyzer:

    def __init__(self):

        self.low_focus_threshold = 40
        self.blink_fatigue_threshold = 25

        self.break_minutes = 15
        self.drop_window = 5


    def load_data(self):

        if not Path(DATA_PATH).exists():
            return None

        df = pd.read_csv(DATA_PATH)

        if df.empty:
            return None

        return df


    def last_minutes(self, df, minutes):

        now = time.time()
        cutoff = now - (minutes * 60)

        return df[df["timestamp"] >= cutoff]


    def check_low_focus(self, df):

        recent = self.last_minutes(df, self.break_minutes)

        if recent.empty:
            return None

        avg_focus = recent["focus_score"].mean()

        if avg_focus < self.low_focus_threshold:

            return {
                "type": "LOW_FOCUS",
                "message": "Your focus has been low for 15 minutes. Consider taking a short break.",
                "avg_focus": round(avg_focus, 2)
            }

        return None


    def check_focus_drop(self, df):

        recent = self.last_minutes(df, self.drop_window)

        if len(recent) < 5:
            return None

        start_focus = recent["focus_score"].iloc[0]
        end_focus = recent["focus_score"].iloc[-1]

        drop = start_focus - end_focus

        if drop > 30:

            return {
                "type": "FOCUS_DROP",
                "message": "Your focus dropped significantly in the last few minutes.",
                "drop": round(drop, 2)
            }

        return None


    def check_fatigue(self, df):

        recent = self.last_minutes(df, 5)

        if recent.empty:
            return None

        avg_blink = recent["blink_rate"].mean()
        avg_focus = recent["focus_score"].mean()

        if avg_blink > self.blink_fatigue_threshold and avg_focus < 50:

            return {
                "type": "FATIGUE",
                "message": "High blink rate detected. You might be getting tired.",
                "blink_rate": round(avg_blink, 2)
            }

        return None


    def analyze(self):

        df = self.load_data()

        if df is None:
            return []

        alerts = []

        for check in [
            self.check_low_focus,
            self.check_focus_drop,
            self.check_fatigue
        ]:

            result = check(df)

            if result:
                alerts.append(result)

        return alerts


if __name__ == "__main__":

    analyzer = FocusAnalyzer()

    while True:

        alerts = analyzer.analyze()

        for alert in alerts:

            print(f"[ALERT] {alert['type']}: {alert['message']}")

        time.sleep(10)