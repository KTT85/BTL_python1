import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Đọc dữ liệu
df = pd.read_csv('transfer_values.csv')
print(f"Số lượng dữ liệu ban đầu: {len(df)}")

# Tiền xử lý dữ liệu
def preprocess_data(df):
    print("\nThông tin dữ liệu trước khi xử lý:")
    print(df.info())
    print("\nMẫu dữ liệu ban đầu:")
    print(df.head())

    # Chuyển đổi Transfer_Value từ chuỗi sang float
    df['Transfer_Value'] = df['Transfer_Value'].str.replace('€', '').str.replace('M', '').astype(float)
    print(f"\nSố lượng dữ liệu sau khi chuyển đổi Transfer_Value: {len(df)}")

    # Nhóm tuổi
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 21, 25, 30, 100], labels=['<21', '21-25', '26-30', '30+'])

    # Chuẩn hóa Minutes
    scaler = StandardScaler()
    df['Minutes_Normalized'] = scaler.fit_transform(df[['Minutes']])

    # Kinh nghiệm dựa theo số phút thi đấu
    df['Experience_Level'] = pd.cut(df['Minutes'], bins=[0, 1000, 2000, 3000, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])

    # Phân loại Position
    position_map = {
        'GK': 'GK',
        'DF': 'DEF',
        'MF': 'MID',
        'FW': 'FWD'
    }
    df['Position_Type'] = df['Position'].map(lambda x: position_map.get(x[:2], 'Other'))

    print(f"\nSố lượng dữ liệu sau khi tạo Position_Type: {len(df)}")

    print("\nSố lượng giá trị NaN trong từng cột:")
    print(df.isnull().sum())

    df = df.dropna()
    print(f"\nSố lượng dữ liệu sau khi xóa các hàng có NaN: {len(df)}")

    print("\nThông tin dữ liệu sau khi xử lý:")
    print(df.info())
    print("\nMẫu dữ liệu sau khi xử lý:")
    print(df[['Player', 'Age', 'Position_Type', 'Transfer_Value']].head())

    return df

# Hàm đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\nRMSE: {rmse:.2f}M")
    print(f"R2 Score: {r2:.2f}")

    return y_test, y_pred, rmse

# Hàm khuyến nghị chuyển nhượng
def recommend_transfers(df, y_pred):
    df_result = df.copy()
    df_result['Predicted_Value'] = y_pred
    df_result['Difference'] = df_result['Predicted_Value'] - df_result['Transfer_Value']

    print("\nTop 10 cầu thủ nên mua (giá trị dự đoán cao hơn giá thị trường):")
    print(df_result.sort_values(by='Difference', ascending=False).head(10)[['Player', 'Team', 'Transfer_Value', 'Predicted_Value', 'Difference']])

    print("\nTop 10 cầu thủ nên bán (giá trị dự đoán thấp hơn giá thị trường):")
    print(df_result.sort_values(by='Difference').head(10)[['Player', 'Team', 'Transfer_Value', 'Predicted_Value', 'Difference']])

    # Lưu ra file nếu cần
    df_result.to_csv('player_transfer_recommendations.csv', index=False)

# Xử lý dữ liệu
df = preprocess_data(df)

# Chuẩn bị dữ liệu huấn luyện
features = ['Minutes_Normalized', 'Age', 'Age_Group', 'Experience_Level', 'Position_Type']
X = df[features].copy()
y = df['Transfer_Value']

le = LabelEncoder()
X['Position_Type'] = le.fit_transform(X['Position_Type'])
X['Age_Group'] = le.fit_transform(X['Age_Group'])
X['Experience_Level'] = le.fit_transform(X['Experience_Level'])

print(f"\nKích thước của tập features (X): {X.shape}")
print(f"Kích thước của tập target (y): {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nKích thước tập train: {X_train.shape}")
print(f"Kích thước tập test: {X_test.shape}")

# Huấn luyện mô hình
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá mô hình
y_test, y_pred, rmse = evaluate_model(model, X_test, y_test)

# Dự đoán toàn bộ cầu thủ
y_all_pred = model.predict(X)

# Biểu đồ + Khuyến nghị vào file PDF
with PdfPages("player_value_report.pdf") as pdf:
    # Biểu đồ RMSE
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Giá trị thực tế (M EUR)")
    plt.ylabel("Giá trị dự đoán (M EUR)")
    plt.title(f"Biểu đồ dự đoán giá trị cầu thủ\nRMSE: {rmse:.2f}M")
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title("Feature Importance")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# Đưa ra khuyến nghị chuyển nhượng trên toàn bộ cầu thủ
recommend_transfers(df, y_all_pred)
print("\n Đã lưu biểu đồ và khuyến nghị cho toàn bộ cầu thủ vào player_value_report.pdf và player_transfer_recommendations.csv")
