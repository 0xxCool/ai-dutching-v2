import pandas as pd
from sportmonks_dutching_system import TeamMatcher

# Lade Datenbank
df = pd.read_csv('game_database_complete.csv')

# Normalisiere Team-Namen
df['home_team'] = df['home_team'].apply(TeamMatcher.normalize)
df['away_team'] = df['away_team'].apply(TeamMatcher.normalize)

# Speichere
df.to_csv('game_database_complete_normalized.csv', index=False)

print("✅ Datenbank normalisiert!")