import json
import numpy as np
import pandas as pd

def analyze():
    df = pd.read_csv('oof_predictions.csv')
    with open('best_threshold.json') as f:
        threshold = json.load(f)['threshold']

    df['pred'] = (df['oof_prob_class1'] >= threshold).astype(int)
    df['correct'] = df['pred'] == df['label']
    df['confidence'] = np.where(df['pred'] == 1, df['oof_prob_class1'], 1 - df['oof_prob_class1'])

    wrong = df[~df['correct']].sort_values('confidence', ascending=False)
    print(f"Total misclassified: {len(wrong)} / {len(df)} ({len(wrong)/len(df)*100:.1f}%)")

    cols = ['image_id', 'sun_azimuth_angle', 'label', 'pred', 'confidence']
    print("\nTop 20 most CONFIDENTLY WRONG (highest priority to open and look at):")
    print(wrong[cols].head(20).to_string(index=False))

    wrong.to_csv('misclassified_examples.csv', index=False)
    print("\nFull list saved to misclassified_examples.csv.")
    print("Open a handful of the top rows' images and look for a common pattern:")
    print("  - clustered around a specific azimuth range (grazing light)?")
    print("  - low-contrast / flat-looking scenes (weak shadows to begin with)?")
    print("  - a particular feature SIZE (tiny rocks vs huge domes)?")
    print("Whatever pattern shows up there is your best clue for what to fix next.")

if __name__ == '__main__':
    analyze()