import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
data_csv=pd.read_csv('Iris.csv')
iris_df=pd.DataFrame(data_csv)
iris_df
iris_df['Species'] = iris_df['Species'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
iris_df
X = iris_df.drop(columns=['Species','Id'])  # Features
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

# Calculate mean
means = X[numeric_columns].mean()
std_devs=X[numeric_columns].std()
# Centered the data by subtracting the mean and dividing by standard deviation
X_standardized = (X[numeric_columns] - means) / std_devs
X_standardized
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.spatial.distance import euclidean
y = iris_df['Species']  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=20)
# Print results
print("Original Features:")
print(X[:5])
y_train=y_train.reset_index(drop=True)
print("\nStandardized Features:")
print(X_standardized[:5])

def normal_KNN(k, X_train, X_test, y_train, y_test):
  all_top_k_indices =[]
  distance_matrix=[]
  # Iterate over each point in X_test
  for x_test_point in X_test.values:

    distances = [euclidean(x_test_point, x_train) for x_train in X_train.values]
    # Get indices of k-nearest training data points
    k_neighbors_indices = np.argsort(distances)[:k]
    k_neighbours_distances=np.sort(distances)[:k]
    species_matrix= y_train.loc[k_neighbors_indices].values

    #Saving values
    all_top_k_indices.append(species_matrix)
    distance_matrix.append(k_neighbours_distances)

  # Converting the list to a NumPy array for convenience
  all_top_k_indices_array = np.array(all_top_k_indices)
  distance_matrix_array = np.array(distance_matrix)
  y_pred_normal = np.array([np.bincount(row).argmax() for row in all_top_k_indices_array])
  
  y_pred_weighted=weighted_KNN(all_top_k_indices_array,distance_matrix_array,k,X_train, X_test, y_train, y_test)
  return [y_pred_normal,y_pred_weighted]

def weighted_KNN(all_top_k_indices_array,distance_matrix_array,k,X_train, X_test, y_train, y_test):
  # Initialize an array to store the results
  num_rows, num_cols = all_top_k_indices_array.shape
  result_matrix = np.zeros((num_rows, 3))  # 3 columns for 0s, 1s, and 2s

  # Iterate over each row
  for i in range(num_rows):
    # Find indices where all_top_k_indices_array contains 0, 1, and 2
    indices_0 = np.where(all_top_k_indices_array[i] == 0)[0]
    indices_1 = np.where(all_top_k_indices_array[i] == 1)[0]
    indices_2 = np.where(all_top_k_indices_array[i] == 2)[0]

    # Sum distances corresponding to 0s, 1s, and 2s
    sum_0 = np.sum(distance_matrix_array[i, indices_0]) if indices_0.size > 0 else 0
    sum_1 = np.sum(distance_matrix_array[i, indices_1]) if indices_1.size > 0 else 0
    sum_2 = np.sum(distance_matrix_array[i, indices_2]) if indices_2.size > 0 else 0

    # Store the results in the result_matrix
    result_matrix[i] = [sum_0, sum_1, sum_2]

  # Evaluate the model accuracy
  y_pred = np.argmax(result_matrix, axis=1)
  return y_pred

k_values=[1, 3, 5, 10, 20]
accuracy_matrix=[]
y_pred_comb=[]

for k in k_values:
  y_pred_init=normal_KNN(k, X_train, X_test, y_train, y_test)
  y_pred_comb.append(y_pred_init)

  y_test.reset_index(drop=True,inplace=True)
  y_test_array=np.array(y_test)
  #Accuracy of Normal KNN
  matching_elements = y_test_array == y_pred_init[0]
  num_matching_elements = np.sum(matching_elements)
  accuracy_normal=num_matching_elements/len(y_pred_init[0])
  print(f"Accuracy of normal K-NN (KNN_Normal) is : {accuracy_normal:.2f} for K={k}")
  #Accuracy of Weighted KNN
  matching_elements = y_test_array == y_pred_init[1]
  num_matching_elements = np.sum(matching_elements)
  accuracy_weighted=num_matching_elements/len(y_pred_init[1])
  print(f"Accuracy of distance weighted K-NN (KNN_Weighted) is : {accuracy_weighted:.2f} for K={k}")

  accuracy_matrix.append([accuracy_normal,accuracy_weighted])


accuracy_matrix_array=np.array(accuracy_matrix)
y_pred=np.array(y_pred_comb)
#EXPERIMENT 1 of normal_KNN
from sklearn.metrics import ConfusionMatrixDisplay
# Plot Percentage Accuracy vs K
plt.plot(k_values, accuracy_matrix_array[:, 0], marker='o')
plt.title('Percentage Accuracy vs K(KNN_Normal)')
plt.xlabel('K')
plt.ylabel('Percentage Accuracy')
plt.show()

# Find the best value of the hyperparameter K
best_k_index = np.argmax(accuracy_matrix_array[:, 0])
best_k = k_values[best_k_index]
print(f"Best value of K: {best_k} with accuracy: {accuracy_matrix_array[best_k_index, 0]:.2%}")

# Plot Confusion Matrix for the best K
best_k_confusion_matrix = confusion_matrix(y_test, y_pred[best_k_index][0])
display = ConfusionMatrixDisplay(confusion_matrix=best_k_confusion_matrix, display_labels=np.unique(y_test))
display.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix for best K (K={best_k}->KNN_Normal)')
plt.show()

#EXPERIMENT 2 of weighted_KNN
# Plot Percentage Accuracy vs K
plt.plot(k_values, accuracy_matrix_array[:, 1], marker='o', color='red')
plt.title('Percentage Accuracy vs K(KNN_Weighted)')
plt.xlabel('K')
plt.ylabel('Percentage Accuracy')
plt.show()

# Find the best value of the hyperparameter K
best_k_index = np.argmax(accuracy_matrix_array[:, 1])
best_k = k_values[best_k_index]
print(f"Best value of K: {best_k} with accuracy: {accuracy_matrix_array[best_k_index, 1]:.2%}")

# Plot Confusion Matrix for the best K
best_k_confusion_matrix = confusion_matrix(y_test, y_pred[best_k_index][1])
display = ConfusionMatrixDisplay(confusion_matrix=best_k_confusion_matrix, display_labels=np.unique(y_test))
display.plot(cmap='Reds', values_format='d')
plt.title(f'Confusion Matrix for best K (K={best_k}->KNN_Weighted)')
plt.show()