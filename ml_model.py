import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Generate Mock Dataset
# Features: Math_Score, Science_Score, Communication_Score, Creativity_Score, Programming_Score (out of 10)
# Target: 0 = Engineering, 1 = Medicine, 2 = Arts, 3 = Commerce/Business
data = []
for _ in range(500):
    career = np.random.choice([0, 1, 2, 3])
    if career == 0:
        # Engineering
        math = np.random.randint(7, 11)
        sci = np.random.randint(7, 11)
        comm = np.random.randint(4, 9)
        creative = np.random.randint(4, 9)
        prog = np.random.randint(7, 11)
    elif career == 1:
        # Medicine
        math = np.random.randint(6, 10)
        sci = np.random.randint(8, 11)
        comm = np.random.randint(6, 10)
        creative = np.random.randint(4, 8)
        prog = np.random.randint(2, 6)
    elif career == 2:
        # Arts
        math = np.random.randint(3, 8)
        sci = np.random.randint(3, 8)
        comm = np.random.randint(8, 11)
        creative = np.random.randint(8, 11)
        prog = np.random.randint(1, 5)
    else:
        # Commerce/Business
        math = np.random.randint(7, 11)
        sci = np.random.randint(4, 8)
        comm = np.random.randint(7, 11)
        creative = np.random.randint(5, 9)
        prog = np.random.randint(3, 7)
    
    data.append([math, sci, comm, creative, prog, career])

df = pd.DataFrame(data, columns=['Math', 'Science', 'Communication', 'Creativity', 'Programming', 'Career'])
X = df.drop('Career', axis=1)
y = df['Career']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
model_path = os.path.join(os.path.dirname(__file__), 'career_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path} with accuracy: {model.score(X_test, y_test)}")
