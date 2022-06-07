import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data1 = pd.read_csv("heart.csv")

sns.heatmap(data1.corr()[['DEATH_EVENT']].sort_values(by='DEATH_EVENT', ascending=False), annot = True)
plt.show()

sns.heatmap(data1.corr(), annot = True)
plt.rcParams["figure.figsize"] = (15,15)
plt.show()

sns.scatterplot(x=data1["serum_creatinine"], y=data1["ejection_fraction"], hue=data1["DEATH_EVENT"])
plt.show()

sns.scatterplot(x=data1["serum_sodium"], y=data1["ejection_fraction"], hue=data1["DEATH_EVENT"])
plt.show()

sns.barplot(x=data1["DEATH_EVENT"], y=data1["smoking"], hue=data1["sex"])
plt.show()

#SMOKING

none = 0
both = 0
noDyesS = 0
noSyesD = 0

for x in range(100, len(data1["smoking"])):
    if data1.at[x, "smoking"] == 0 and data1.at[x, "DEATH_EVENT"] == 0:
        none += 1
    if data1.at[x, "smoking"] == 1 and data1.at[x, "DEATH_EVENT"] == 1:
        both += 1
    if data1.at[x, "smoking"] == 1 and data1.at[x, "DEATH_EVENT"] == 0:
        noDyesS += 1
    if data1.at[x, "smoking"] == 0 and data1.at[x, "DEATH_EVENT"] == 1:
        noSyesD += 1

sns.barplot(x=["none", "both", "smoke+!death", "death+!smoke"], y=[none, both, noDyesS, noSyesD])

plt.show()

print("Death Due to Smoking", " P(Death) =", both / len(data1["smoking"]))

#ANAEMIA

none = 0
both = 0
noDyesS = 0
noSyesD = 0

for x in range(100, len(data1["anaemia"])):
    if data1.at[x, "anaemia"] == 0 and data1.at[x, "DEATH_EVENT"] == 0:
        none += 1
    if data1.at[x, "anaemia"] == 1 and data1.at[x, "DEATH_EVENT"] == 1:
        both += 1
    if data1.at[x, "anaemia"] == 1 and data1.at[x, "DEATH_EVENT"] == 0:
        noDyesS += 1
    if data1.at[x, "anaemia"] == 0 and data1.at[x, "DEATH_EVENT"] == 1:
        noSyesD += 1

sns.barplot(x = ["none", "both", "anaemia+!death", "death+!anaemia"], y = [none, both, noDyesS, noSyesD])

plt.show()
print("Death Due to Anaemia", " P(Death) =", both/len(data1["anaemia"]))
print("Those Who have Anaemia", " P(Anaemia) =", noDyesS/len(data1["anaemia"]))

#DIABETES

none = 0
both = 0
noDyesS = 0
noSyesD = 0

for x in range(100, len(data1["diabetes"])):
    if data1.at[x, "diabetes"] == 0 and data1.at[x, "DEATH_EVENT"] == 0:
        none += 1
    if data1.at[x, "diabetes"] == 1 and data1.at[x, "DEATH_EVENT"] == 1:
        both += 1
    if data1.at[x, "diabetes"] == 1 and data1.at[x, "DEATH_EVENT"] == 0:
        noDyesS += 1
    if data1.at[x, "diabetes"] == 0 and data1.at[x, "DEATH_EVENT"] == 1:
        noSyesD += 1
        sns.barplot(x=["none", "both", "diabetes+!death", "death+!diabetes"], y=[none, both, noDyesS, noSyesD])

        plt.show()

        print("Death Due to Diabetes", " P(Death) =", both / len(data1["diabetes"]))
        print("Those Who have Anaemia", " P(Anaemia) =", noDyesS / len(data1["diabetes"]))

        # AGE and DEATH

        sns.histplot(data=data1, x="age", kde=True, hue="DEATH_EVENT")
        plt.show()

        # AGE, TIME and DEATH EVENT

        sns.kdeplot(x="time", y="age", hue="DEATH_EVENT", data=data1, pallete="CMRmap")
        plt.show()

        # all plot of variable correlations

        sns.pairplot(data1, hue="DEATH_EVENT", palette="CMRmap")
        plt.show()

        f1 = []

        for x in range(0, len(data1["sex"])):
            if data1["sex"].iat[x] == 1 and data1["smoking"].iat[x] == 1:
                f1.append(1)
            else:
                f1.append(0)

        data1.insert(loc=0, column='male_smoker', value=f1)

        data = data1.iloc[50:, ]
        data2 = data1.iloc[:50, ]

        realval = data2["DEATH_EVENT"]
        data2.drop("DEATH_EVENT", axis=1, inplace=True)

        # EMPTY COLUMNS

        nullCol = []

        for x in data:
            z = 0
            for y in pd.isnull(data[x]):
                if y == True:
                    z += 1
            if z != 0:
                nullCol.append(x)

                from sklearn.preprocessing import MinMaxScaler


                def normalize(data):
                    scaler = MinMaxScaler()
                    data["creatinine_phosphokinase"] = scaler.fit_transform(
                        data["creatinine_phosphokinase"].values.reshape(-1, 1))
                    data["platelets"] = scaler.fit_transform(data["platelets"].values.reshape(-1, 1))
                    data["serum_sodium"] = scaler.fit_transform(data["serum_sodium"].values.reshape(-1, 1))
                    return data


                normData = normalize(data)
                normData2 = normalize(data2)

                import sklearn.model_selection as sk

                targets = normData["DEATH_EVENT"]

                normData.drop('DEATH_EVENT', axis=1, inplace=True)

                X_train, X_val, Y_train, Y_val = sk.train_test_split(normData, targets, test_size=0.2, random_state=42,
                                                                     shuffle=True)

                from sklearn.model_selection import train_test_split
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.svm import SVC
                from xgboost import XGBClassifier
                from sklearn import metrics
                from sklearn.naive_bayes import GaussianNB
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

                tree = DecisionTreeClassifier(random_state=24)
                forest = RandomForestClassifier(random_state=24)
                knn = KNeighborsClassifier()
                svm = SVC(random_state=24)
                xboost = XGBClassifier(random_state=24)
                gauss = GaussianNB()
                LDA = LinearDiscriminantAnalysis()

                models = [tree, forest, knn, svm, xboost, gauss, LDA]
                plotData = []
                plotData1 = []
                for model in models:
                    model.fit(X_train, Y_train)
                    y_pred = model.predict(X_val)
                    accuracy = metrics.accuracy_score(Y_val, y_pred)
                    plotData.append(accuracy)
                    plotData1.append(type(model).__name__)
                    # clf_report= metrics.classification_report(Y_val, y_pred)
                    # print(f"The accuracy of model {type(model).__name__} is {accuracy:.2f}")
                    # print(clf_report)
                    # print("\n")

                sns.barplot(y=plotData1, x=plotData)
                plt.show()

                from sklearn.datasets import make_classification
                from sklearn.model_selection import GridSearchCV
                from sklearn.model_selection import RepeatedStratifiedKFold
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                from numpy import arange

                model = LinearDiscriminantAnalysis()

                # Cross validation method, cross validate one set of tunings with all other sets of tunings
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)  # What do the parameters mean?
                grid = dict()
                grid["solver"] = ["svd", "lsqr", "eigen"]
                search = GridSearchCV(model, grid, scoring="accuracy", cv=cv, n_jobs=-1)

                results = search.fit(X_train, Y_train)

                print('Mean Accuracy: %.3f' % results.best_score_)
                print('Config: %s' % results.best_params_)

                from sklearn.datasets import make_classification
                from sklearn.model_selection import GridSearchCV
                from sklearn.model_selection import RepeatedStratifiedKFold
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                from numpy import arange

                model = LinearDiscriminantAnalysis(solver="lsqr")
                # Cross validation method, cross validate one set of tunings with all other sets of tunings
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)  # What do the parameters mean?

                grid = dict()
                grid["shrinkage"] = arange(0, 1, 0.01)
                search = GridSearchCV(model, grid, scoring="accuracy", cv=cv, n_jobs=-1)

                results = search.fit(X_train, Y_train)
                print('Mean Accuracy: %.3f' % results.best_score_)
                print('Config: %s' % results.best_params_)

                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

                clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.42)

                clf.fit(X_train, Y_train)

                y_pred = clf.predict(X_val)

                print(y_pred)

                from sklearn import metrics

                print("Accuracy:", metrics.accuracy_score(Y_val, y_pred))

                # Val data

                TP = 0
                TN = 0
                FN = 0
                FP = 0

                for x in range(0, len(list(y_pred)) - 1):
                    if list(Y_val)[x] == 1 and list(y_pred)[x] == 1:
                        TP += 1
                    if list(Y_val)[x] == 0 and list(y_pred)[x] == 0:
                        TN += 1
                    if list(Y_val)[x] == 1 and list(y_pred)[x] == 0:
                        FN += 1
                    if list(Y_val)[x] == 0 and list(y_pred)[x] == 1:
                        FP += 1
print("TRUE POSITIVE: ", TP/len(list(realval)))
print("---------------------------------------")
print("TRUE NEGATIVE: ", TN/len(list(realval)))
print("---------------------------------------")
print("FALSE POSITIVE: ", FP/len(list(realval)))
print("---------------------------------------")
print("FALSE NEGATIVE: ", FN/len(list(realval)))

a = list(y_pred)
ax = sns.countplot(x=a)
ax.bar_label(ax.containers[0])
plt.xlabel("Predicted Count of deaths (1) and no deaths (0)")
plt.show()

preds = clf.predict(normData2)

#RANDOM DATA

TP = 0
TN = 0

FN = 0
FP = 0

for x in range(0, len(list(realval))-1):
    if list(realval)[x] == 1 and list(preds)[x] == 1:
        TP += 1
    elif list(realval)[x] == 0 and list(preds)[x] == 0:
        TN += 1
        # elif not list(realval)[x] != 1 and list(preds)[x] == 0:
        # FN += 1
    elif list(realval)[x] == 0 and list(preds)[x] == 1:
        FP += 1

print("TRUE POSITIVE: ", TP / len(list(realval)))
print("---------------------------------------")
print("TRUE NEGATIVE: ", TN / len(list(realval)))
print("---------------------------------------")
print("FALSE POSITIVE: ", FP / len(list(realval)))
print("---------------------------------------")
print("FALSE NEGATIVE: ", FN / len(list(realval)))
a = list(preds)
ax = sns.countplot(x=a)
ax.bar_label(ax.containers[0])
plt.xlabel("Predicted Count of deaths (1) and no deaths (0)")
plt.show()
a = list(realval)
ax = sns.countplot(x=a)
ax.bar_label(ax.containers[0])
plt.xlabel("Real Count of deaths (1) and no deaths (0)")
plt.show()
