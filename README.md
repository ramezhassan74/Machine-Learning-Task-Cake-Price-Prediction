🎯 Cake Price Prediction Project

Scroll for Arabic ⬇️

📘 Overview

This project was created as part of the Machine Learning Engineer Virtual Internship (Intern2Grow).
The goal was to build a machine learning model that can predict the price of a cake based on various features like size, ingredients cost, design complexity, and more.

🧠 Steps and Methodology
1️⃣ Data Preparation

The dataset contained categorical columns such as Sold_On, Size, Design_Complexity, and Gender.

These were encoded into numerical values using LabelEncoder to make them suitable for the machine learning model.

Outliers were also removed by filtering rows below the 95th percentile of the Price column to improve accuracy.

2️⃣ Feature Selection

The features used for training were:
Sold_On, Size, Ingredients_Cost, Design_Complexity, Time_Taken, Amount, and Gender.

The target variable (the one we want to predict) was Price.

3️⃣ Model Training

A Random Forest Regressor was chosen because it’s a strong, flexible model that performs well on complex datasets.

Initially, a simple Random Forest was trained to establish a baseline performance.

4️⃣ Model Optimization

To improve results, GridSearchCV was used to test different combinations of hyperparameters:

n_estimators: number of trees

max_depth: depth of each tree

min_samples_split: minimum samples required to split a node

The model with the lowest Mean Absolute Error (MAE) was selected.

5️⃣ Model Evaluation

Two metrics were used to evaluate the model’s performance:

MAE (Mean Absolute Error): ≈ 5.68

RMSE (Root Mean Squared Error): ≈ 7.39

These low error values indicate that the model is very accurate for this dataset.

6️⃣ Price Prediction

Finally, the model was used to predict the price of a new cake with custom input features.

Example prediction result:

Predicted Price: 72.25

🧩 Tools and Libraries

Python 3.11

pandas for data handling

scikit-learn for modeling, training, and evaluation

numpy for mathematical operations

📊 Conclusion

The project successfully built a predictive model that estimates cake prices with high accuracy.
Further improvements could include:

Using OneHotEncoding for categorical data

Performing feature scaling

Trying advanced ensemble methods like Gradient Boosting or XGBoost

🇪🇬 (Scroll for Arabic)
🎯 مشروع التنبؤ بسعر الكيك

المشروع ده جزء من برنامج التدريب الافتراضي لمهندس تعلم الآلة (Intern2Grow)، والهدف منه إننا نبني موديل تعلم آلي يقدر يتنبأ بسعر الكيكة بناءً على شوية عوامل زي الحجم، تكلفة المكونات، درجة التعقيد في التصميم، والوقت اللي استغرقه الشيف في التحضير.

🧠 خطوات التنفيذ
1️⃣ تجهيز البيانات

الأعمدة اللي فيها بيانات نصية (زي الأيام أو النوع أو التعقيد) اتحولت لأرقام باستخدام LabelEncoder.

تم كمان حذف القيم الشاذة (Outliers) اللي بتأثر على دقة الموديل.

2️⃣ اختيار الخصائص (Features)

الخصائص اللي استخدمناها في التدريب:
Sold_On, Size, Ingredients_Cost, Design_Complexity, Time_Taken, Amount, Gender
أما العمود اللي بنتنبأ بيه هو Price.

3️⃣ تدريب الموديل

استخدمنا Random Forest Regressor لأنه قوي وسهل التعامل معاه.

دربناه في البداية على شكل بسيط علشان نعرف الأداء المبدئي.

4️⃣ تحسين الأداء

استخدمنا GridSearchCV علشان نجرب أكتر من إعداد (parameters) للموديل زي:

عدد الأشجار n_estimators

عمق الشجرة max_depth

أقل عدد عينات للتقسيم min_samples_split

واخترنا الإعدادات اللي جابت أقل خطأ.

5️⃣ التقييم

النتائج كانت ممتازة:

MAE = 5.68

RMSE = 7.39
وده معناه إن الموديل دقيق جدًا في التنبؤ.

6️⃣ التنبؤ بسعر جديد

جربنا ندخل بيانات لكيكة جديدة، والموديل تنبأ إن سعرها حوالي 72.25.

⚙️ الأدوات

Python

pandas

scikit-learn

numpy

🏁 النتيجة

قدرنا نبني موديل تعلم آلي دقيق جدًا في التنبؤ بأسعار الكيك.
ولو حبيت تطوره أكتر، ممكن تستخدم OneHotEncoding أو Gradient Boosting لتحسين النتائج أكتر.
