import numpy as np
import pandas as pd
import sklearn.svm
from sklearn.model_selection import train_test_split


def classification_accuracy(labels, predictions):
    correct = np.where(labels == predictions)[0]
    accuracy = correct.shape[0]/labels.shape[0]
    return accuracy



def fitness(sol, total_features, label, split):
  feature =  reduce_features(sol, total_features)
  xtrain, xtest, ytrain, ytest = train_test_split(feature, label, test_size = split, random_state = 4 )
  SVM_classifier = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5)
  SVM_classifier.fit(xtrain, ytrain)
  predictions = SVM_classifier.predict(xtest)
  fitness_value = classification_accuracy(ytest, predictions)

  return fitness_value



def hamming_distance(b1,b2):
  ans = 0
  for i in range(len(b1)):
    ans += not(b1[i]==b2[i])

  return ans


def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features



def similarity(beta, H_d, A1, A2):
  D_a = abs(A1-A2)
  if (H_d !=0):
    if (D_a !=0): 
      S = beta/H_d + (1-beta)/D_a
    else :
      S = 99999
  else :
    S = 99999
  return S



f = '/content/drive/My Drive/HErlev_GoogLeNet_b_original_.csv'
dataframe = pd.read_csv(f)
data_inputs = np.array(dataframe)

df= pd.read_csv('/content/drive/My Drive/HErlev_Class.csv')
class_df=df['Class']
data_outputs = np.asarray(class_df)

num_feature_elements = data_inputs.shape[1]

sol_per_pop = 10
n_clusters = 5


pop_shape = (sol_per_pop, num_feature_elements)
cluster_shape = (n_clusters, num_feature_elements)

new_population = np.random.randint(low=0, high=2, size=pop_shape)
cluster_centres = np.random.randint(low=0, high=2, size=cluster_shape)

clusters_population = []
cluster_similarities = []
for j in range(n_clusters):
  temp_cluster = []
  temp_similarity = []
  temp_cluster.append(cluster_centres[j])
  temp_similarity.append(1e4)
  for i in range(len(cluster_similarity)):
    max_idx = cluster_similarity[i].index(max(cluster_similarity[i]))
    print('population: ',i)
    print('index: ',max_idx)
    if (max_idx == j):
      temp_cluster.append(new_population[i])
      temp_similarity.append(cluster_similarity[i][max_idx])
  
  clusters_population.append(temp_cluster)
  cluster_similarities.append(temp_similarity)
beta = 0.3
cluster_similarity = []
for i in range(len(new_population)):
  new_sol = new_population[i]
  acc_sol = fitness(new_sol, data_inputs, data_outputs, 0.5)
  similarity_list = []
  for j in range(len(cluster_centres)):
    print('Population: ', i)
    print('Cluster: ', j)
    C_cluster = cluster_centres[j]
    acc_cluster = fitness(C_cluster, data_inputs, data_outputs, 0.5)
    H_distance = hamming_distance(C_cluster, new_sol)
    Similarity_score = similarity(beta, H_distance, acc_sol, acc_cluster)
    similarity_list.append(Similarity_score)
    print('Similarity_score: ', Similarity_score)
  cluster_similarity.append(similarity_list)
  
  weighted_solution = []
for i in range(n_clusters):
  temp_weighted_solution = []
  for j in range(len(clusters_population[i])):
    candidate_solution = clusters_population[i][j]
    Acc_d = fitness(candidate_solution, data_inputs, data_outputs, 0.5)
    temp_goodness = []
    for k in range(len(clusters_population[i][j])):
      temp_goodness.append(clusters_population[i][j][k]*Acc_d)
    temp_weighted_solution.append(temp_goodness)
  weighted_solution.append(temp_weighted_solution)

summed_solution = []
for i in range(n_clusters):
  for j in range(len(weighted_solution[i])):
    if (j==0):
      sum_list = weighted_solution[i][j]
    else:
      sum_list = [(a + b) for a, b in zip(sum_list, weighted_solution[i][j])]
  summed_solution.append(sum_list)

h_d_thresh = []
for i in range(len(summed_solution)):
  sum = 0
  for j in range(len(summed_solution[i])):
    sum += summed_solution[i][j]/len(summed_solution[i])
  h_d_thresh.append(sum)
  
  final_population = []
for i in range(len(h_d_thresh)):
  final_solution = []
  for j in range(len(summed_solution[i])):
    if (summed_solution[i][j] >= h_d_thresh[i]):
      final_solution.append(1)
    else:
      final_solution.append(0)
  final_population.append(final_solution)
  initial_pop_from_cluster = pd.DataFrame(final_population)
  initial_pop_from_cluster.to_csv('initial_population.csv')
