import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from torchmetrics import Accuracy, Precision, Recall, F1Score, MeanSquaredError
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import optuna
from optuna.integration import PyTorchLightningPruningCallback


# Load the data:
train = pd.read_csv('C:/Users/julie/Downloads/train.csv')
test = pd.read_csv('C:/Users/julie/Downloads/test.csv')
sample_submission = pd.read_csv('C:/Users/julie/Downloads/sample_submission.csv')


test_with_sale_price = test.merge(sample_submission, on='Id', how='left')
data = pd.concat([train, test_with_sale_price])

data.drop("Id", axis=1, inplace=True)

#### House Category:
# Function to determine the building type category
def get_type_category(bldg_type):
    if bldg_type in ['1Fam', 'TwnhsE', 'TwnhsI']:
        return 'Residential'
    elif bldg_type in ['Duplex', '2fmCon']:
        return 'Multi-Unit'
    else:
        return 'Special'

# Function to determine the house style category
def get_house_style_category(house_style):
    if house_style in ['1Story', '1.5Fin', '1.5Unf', '2Story']:
        return 'Standard'
    elif house_style in ['2.5Fin', '2.5Unf', 'SFoyer', 'SLvl']:
        return 'Complex'
    else:
        return 'Other'

# Function to determine the age category
def age_category(year_built, year_remod):
    current_year = 2024
    if current_year - year_remod <= 20 or current_year - year_built <= 20:
        return 'New'
    else:
        return 'Older'

# Apply functions to create new columns
data['Age Category'] = data.apply(lambda x: age_category(x['YearBuilt'], x['YearRemodAdd']), axis=1)
data['Type Category'] = data.apply(lambda x: get_type_category(x['BldgType']), axis=1)
data['House Style Category'] = data.apply(lambda x: get_house_style_category(x['HouseStyle']), axis=1)

# Function to combine age, type, and style categories to create a 'Composite House Category'
def get_final_composite_house_category(row):
    # Creating six distinct categories by combining elements
    if row['Type Category'] == 'Residential':
        return f"{row['Age Category']} Residential {row['House Style Category']}"
    elif row['Type Category'] == 'Multi-Unit':
        return f"{row['Age Category']} Multi-Unit {row['House Style Category']}"
    else:
        return f"{row['Age Category']} Special"

# Apply the final composite category function
data['Composite House Category'] = data.apply(get_final_composite_house_category, axis=1)


#Drop columsn related to House Category
data.drop("YearBuilt", axis=1, inplace=True)
data.drop("YearRemodAdd", axis=1, inplace=True)
data.drop("BldgType", axis=1, inplace=True)
data.drop("HouseStyle", axis=1, inplace=True)
data.drop("Age Category", axis=1, inplace=True)
data.drop("Type Category", axis=1, inplace=True)
data.drop("House Style Category", axis=1, inplace=True)


#### EDA ####

### Missing Values 
# Check for missing values in each column
missing_values_count = data.isnull().sum()

sorted_missing_values = missing_values_count.sort_values(ascending=False)

print("Columns with the number of missing values:")
print(sorted_missing_values)

data.drop("PoolQC", axis=1, inplace=True)
data.drop("MiscFeature", axis=1, inplace=True)
data.drop("Alley", axis=1, inplace=True)
data.drop("FireplaceQu", axis=1, inplace=True)

### Removing outliers
sale_price_zscore = zscore(data['SalePrice'])
outlier_mask = np.abs(sale_price_zscore) > 3
outliers = data.loc[outlier_mask, 'SalePrice']
print(f'Num outliers: {len(outliers)}')
data = data.loc[~outlier_mask]


# Impute missing values in numerical variables with mean
numerical_imputer = SimpleImputer(strategy='mean')  
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])

# Impute missing values in categorical variables with most frequent
categorical_imputer = SimpleImputer(strategy='most_frequent')  
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

### Visualization of the data:

# Categorize 'SalePrice' into bins for visualization if it's a continuous variable
data['SalePrice Category'] = pd.cut(data['SalePrice'], bins=[0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, np.inf],
                                    labels=['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k-250k', '250k-300k', '300k-350k', '350k-400k', '400k-450k', '450k-500k', '500k+'])

plt.figure(figsize=(12, 8)) 
sns.countplot(x='SalePrice Category', data=data)
plt.title('Categorical Distribution of Sale Prices')
plt.xlabel('Sale Price Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  
plt.show()

# Pie chart for visualizing the proportion of categories
category_counts = data['SalePrice Category'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal') 
plt.title('Proportion of Sale Price Categories')
plt.show()

# Visualizing imbalance in another categorical variable
plt.figure(figsize=(12, 6))
sns.countplot(x='Composite House Category', data=data)
plt.title('Distribution of House Categories')
plt.xlabel('House Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


data.drop("SalePrice Category", axis=1, inplace=True)

#### Encoding categorical variables: 
label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
for col in categorical_cols:
    data[col] = label_encoders[col].transform(data[col])
    
#### Normalizing numerical values:
# Normalize numerical variables
numeric_cols_except_saleprice = [col for col in data.columns if col != 'SalePrice' and data[col].dtype in ['int64', 'float64']]
scaler = MinMaxScaler()
data[numeric_cols_except_saleprice] = scaler.fit_transform(data[numeric_cols_except_saleprice])

### Feature selction:
from sklearn.feature_selection import SelectKBest, f_regression

k = 20

# Apply feature selection to numerical columns except for target variable 'SalePrice'
feature_selector = SelectKBest(f_regression, k=k)
X_new = feature_selector.fit_transform(data.drop(columns=['SalePrice', 'Composite House Category']), data['SalePrice'])

### Split test/train:
# Split data into features and targets
X_train, X_val, y_train, y_val = train_test_split(X_new, data[['SalePrice', 'Composite House Category']], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

#### Convert to tensors and loaders:
# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_price_tensor = torch.tensor(y_train['SalePrice'].values, dtype=torch.float32)
y_train_category_tensor = torch.tensor(y_train['Composite House Category'].values, dtype=torch.long)
y_val_price_tensor = torch.tensor(y_val['SalePrice'].values, dtype=torch.float32)
y_val_category_tensor = torch.tensor(y_val['Composite House Category'].values, dtype=torch.long)
y_test_price_tensor = torch.tensor(y_test['SalePrice'].values, dtype=torch.float32)
y_test_category_tensor = torch.tensor(y_test['Composite House Category'].values, dtype=torch.long)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_price_tensor, y_train_category_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_price_tensor, y_val_category_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_price_tensor, y_test_category_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


##### Model definitions:

# Define the multi-task model with hyperparameters
class MultiTaskModel(pl.LightningModule):
    def __init__(self, num_features, num_classes, lr, hidden_size, activation, optimizer_name):
        super(MultiTaskModel, self).__init__()
        # Define activations
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activations[activation]
        self.lr = lr
        self.optimizer_name = optimizer_name

        # Increase the depth and complexity of the network
        self.shared_layers = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            self.activation,
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(0.2)
        )

        # Residual connection
        self.residual = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64)
        )

        self.price_head = nn.Sequential(
            nn.Linear(64, 32),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            self.activation,
            nn.Linear(16, 2)  
        )
        
        self.category_head = nn.Sequential(
            nn.Linear(64, 32),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

        self.category_loss_fn = nn.CrossEntropyLoss()
        self.price_loss_fn = self.heteroscedastic_loss

        # Initialize metrics for classification and regression
        self.init_metrics(num_classes)

        # Save all hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        shared_output = self.shared_layers(x)
        residual_output = self.residual(x)
        combined_output = shared_output + residual_output 
        price_output = self.price_head(combined_output)
        price = price_output[:, 0]
        log_var = price_output[:, 1]
        category_output = self.category_head(combined_output)
        return price, log_var, category_output

    def init_metrics(self, num_classes):
        # Initialize metrics for classification
        self.test_accuracy = Accuracy(num_classes=num_classes, average='macro', task='multiclass')
        self.test_precision = Precision(num_classes=num_classes, average='macro', task='multiclass')
        self.test_recall = Recall(num_classes=num_classes, average='macro', task='multiclass')
        self.test_f1_score = F1Score(num_classes=num_classes, average='macro', task='multiclass')

        # Initialize metric for regression
        self.test_rmse = MeanSquaredError(squared=False)

    def heteroscedastic_loss(self, predicted_prices, prices, log_var):
        loss = torch.exp(-log_var) * (predicted_prices - prices) ** 2 + log_var
        return loss.mean()


    def training_step(self, batch, batch_idx):
        features, prices, categories = batch
        predicted_prices, log_var, predicted_categories = self(features)
        price_loss = self.price_loss_fn(predicted_prices, prices, log_var)
        category_loss = self.category_loss_fn(predicted_categories, categories)
        return price_loss + category_loss
    
    def validation_step(self, batch, batch_idx):
        features, prices, categories = batch
        predicted_prices, log_var, predicted_categories = self(features)
        price_loss = self.price_loss_fn(predicted_prices, prices, log_var)
        category_loss = self.category_loss_fn(predicted_categories, categories)
        loss = price_loss + category_loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        features, prices, categories = batch
        predicted_prices, log_var, predicted_categories = self(features)
        price_loss = self.price_loss_fn(predicted_prices, prices, log_var)
        category_loss = self.category_loss_fn(predicted_categories, categories)
        total_loss = price_loss + category_loss

        # Log category-related metrics
        acc = self.test_accuracy(predicted_categories.argmax(dim=1), categories)
        prec = self.test_precision(predicted_categories, categories)
        rec = self.test_recall(predicted_categories, categories)
        f1 = self.test_f1_score(predicted_categories, categories)
    
        # Log price-related metric (RMSE)
        rmse = self.test_rmse(predicted_prices, prices)

        self.log_dict({
            'test_loss': total_loss,
            'test_price_loss': price_loss,
            'test_category_loss': category_loss,
            'test_accuracy': acc,
            'test_precision': prec,
            'test_recall': rec,
            'test_f1_score': f1,
            'test_rmse': rmse
            }, on_step=False, on_epoch=True)

        return {
            'test_loss': total_loss,
            'test_accuracy': acc,
            'test_precision': prec,
            'test_recall': rec,
            'test_f1_score': f1,
            'test_rmse': rmse
            }
    
    def configure_optimizers(self):
        optimizers = {
            'adam': torch.optim.Adam(self.parameters(), lr=self.lr),
            'sgd': torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        }
        optimizer = optimizers[self.optimizer_name]
        return optimizer

# Optuna hyperparameter tuning
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'sigmoid'])
    optimizer_name = trial.suggest_categorical('optimizer_name', ['adam', 'sgd'])
    model = MultiTaskModel(X_train.shape[1], len(np.unique(y_train['Composite House Category'].values)), lr, hidden_size, activation, optimizer_name)

    logger = CSVLogger('logs', name='optuna')
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', save_top_k=1, verbose=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=4,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor, PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        num_sanity_val_steps=0
    )
    trainer.fit(model, train_loader, val_loader)
    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss if val_loss is not None else float('inf')

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10, timeout=600)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
    
    
### Train the model with best parameters:
    
best_params = study.best_trial.params
print("Best trial parameters:", best_params)

# Initialize the model with the best parameters
model = MultiTaskModel(
    num_features=X_train.shape[1], 
    num_classes=len(np.unique(y_train['Composite House Category'].values)), 
    lr=best_params['lr'], 
    hidden_size=best_params['hidden_size'], 
    activation=best_params['activation'], 
    optimizer_name=best_params['optimizer_name']
)

# Setup logging and checkpointing
logger = CSVLogger('logs', name='final_model')
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='best-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)


# Set up the trainer with more epochs for thorough training
trainer = pl.Trainer(
    max_epochs=50,
    logger=logger,
    callbacks=[checkpoint_callback, EarlyStopping(monitor='val_loss', patience=10)],
    accelerator='cpu')

# Train the model
trainer.fit(model, train_loader, val_loader)
# Test the model
trainer.test(model, test_loader)

