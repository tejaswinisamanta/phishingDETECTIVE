#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  # For saving the model & encoders

# Load dataset
df = pd.read_csv("C:/Users/tejas/Downloads/testdata.csv")
df.columns = df.columns.str.strip()

# Define features and target
X = df.drop(columns=['label'])  # Independent variables
y = df['label']  # Target variable (spam or not)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing: One-Hot Encoding for categorical, Scaling for numerical
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Define model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model for future predictions
joblib.dump(model, "spam_detector.pkl")


# In[21]:


df


# In[22]:


# Function to take user input and make predictions
def predict_spam():
    print("\nEnter details for spam detection:")
    
    test_value = {}
    input_prompts = {
        "address type": "('IP addr' or 'DNS name')",
        "url length": "('<54 chars' or 'is 54-75' or '>75 chars')",
        "shortening": "('yes' or 'no')",
        "includes @": "('yes' or 'no')",
        "port number": "('non-std' or 'standard')",
        "domain age": "('< 6 month' or 'older')",
        "redirects": "('<= 4' or '> 4')",
        "domain reg": "('not' or 'expiring')"
    }
    
    for col in X.columns:
        test_value[col] = input(f"Enter value for {col} {input_prompts.get(col, '')}: ").strip()
    
    # Convert input into DataFrame
    test_df = pd.DataFrame([test_value])
    
    # Load trained model
    loaded_model = joblib.load("spam_detector.pkl")
    
    # Make prediction
    prediction = loaded_model.predict(test_df)[0]
    print("\nPrediction: SPAM" if prediction == 1 else "\nPrediction: NOT SPAM")

# Run prediction function
predict_spam()  # Uncomment this to test with user input


# In[24]:


import seaborn as sns

# Plot count distribution for each categorical feature
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[col])
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()


# In[25]:


import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[ ]:




