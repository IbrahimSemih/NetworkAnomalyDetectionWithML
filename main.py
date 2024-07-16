import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# CSV dosyasını yükleme
csv_file = 'C:\\Users\\ibrah\\MachineLearningCVE\\Wednesday-workingHours.pcap_ISCX.csv'
print("Veri yükleniyor...")
df = pd.read_csv(csv_file)

# Sütun adlarını temizleme
df.columns = df.columns.str.strip()

# Etiket sütununu düzenleme
label_column = 'Label'
df['Label'] = df[label_column].apply(lambda x: 'anormal' if x != 'BENIGN' else 'normal')

# Özellik sütunlarını belirleme
feature_columns = ['Destination Port', 'Flow Duration', 'Total Fwd Packets',
                   'Total Backward Packets', 'Total Length of Fwd Packets',
                   'Total Length of Bwd Packets', 'Fwd Packet Length Max',
                   'Fwd Packet Length Min', 'Bwd Packet Length Max',
                   'Bwd Packet Length Min', 'Flow IAT Mean', 'Flow IAT Std',
                   'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
                   'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                   'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
                   'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
                   'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
                   'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
                   'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
                   'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
                   'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                   'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
                   'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
                   'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
                   'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
                   'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
                   'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
                   'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
                   'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean',
                   'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
                   'Idle Std', 'Idle Max', 'Idle Min']

# Özellik ve etiketleri ayırma
X = df[feature_columns]
y = df['Label']

# Eğitim ve test veri setlerine ayırma
print("Veri seti bölünüyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# RandomForest modelini oluşturma ve eğitme
print("Model eğitiliyor...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Eğitim ve test veri setleri üzerinde tahminler
print("Tahminler yapılıyor...")
train_preds = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_preds)
print("Eğitim seti doğruluğu:", train_accuracy)

test_preds = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print("Test seti doğruluğu:", test_accuracy)

print("Sınıflandırma raporu:")
print(classification_report(y_test, test_preds))


# Confusion matrix (karışıklık matrisi) oluşturma ve görselleştirme
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Özellik önemlerini görselleştirme
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feature_columns[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# ROC Curve oluşturma ve görselleştirme
y_test_bin = [1 if label == 'anormal' else 0 for label in y_test]
y_test_scores = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test_bin, y_test_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# Pairplot ile seçilen özellikler arasındaki ilişkiyi görselleştirme
selected_features = ['Flow Duration', 'Total Fwd Packets', 'Total Length of Fwd Packets', 'Fwd Packet Length Max']
subset = df[selected_features + ['Label']]
sns.pairplot(subset, hue='Label', palette={'normal': 'green', 'anormal': 'red'})
plt.show()
