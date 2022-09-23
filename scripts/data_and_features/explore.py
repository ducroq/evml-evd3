import numpy as np
from fetch_data import fetch_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    """feature exploration"""
    data_path = r'G:\My Drive\data\gesture_data'

    # fetch the data
    gestures = fetch_data(data_path)

    # encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(gestures.target)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(gestures.data, coded_labels,
	test_size=0.25, stratify=gestures.target)#, random_state=42)

    # Data preparation (note that a pipeline  would help here)
    trainX = StandardScaler().fit_transform(trainX)

    # show target distribution
    ax = sns.countplot(x=trainY, color="skyblue")
    ax.set_xticklabels(gestures.unique_targets)
    ax.set_title(data_path + ' count')
    plt.tight_layout()

    # show histograms of first 4 features
    fig0, ax0 = plt.subplots(2, 2)
    sns.histplot(trainX[:,0], color="skyblue", bins=10, ax=ax0[0,0])
    sns.histplot(trainX[:,1], color="olive", bins=10, ax=ax0[0,1])#, axlabel=gestures.feature_names[1])
    sns.histplot(trainX[:,2], color="gold", bins=10, ax=ax0[1,0])#, axlabel=gestures.feature_names[2])
    sns.histplot(trainX[:,3], color="teal", bins=10, ax=ax0[1,1])#, axlabel=gestures.feature_names[3])
    ax0[0,0].set_xlabel(gestures.feature_names[0])
    ax0[0,1].set_xlabel(gestures.feature_names[1])
    ax0[1,0].set_xlabel(gestures.feature_names[2])
    ax0[1,1].set_xlabel(gestures.feature_names[3])
    plt.tight_layout()
    
    # show scatter plot of features a and b
    a, b = 0, 1
    fig1 = plt.figure()
    ax1 = sns.scatterplot(x=trainX[:,a], y=trainX[:,b], hue=le.inverse_transform(trainY))
    ax1.set_title("Example of feature scatter plot")
    ax1.set_xlabel(gestures.feature_names[a])
    ax1.set_ylabel(gestures.feature_names[b])
    plt.tight_layout()

##    # show joint distribution plot of features a and b for 2 selected labels
##    a, b = 0, 1
##    c, d = le.transform(['paper', 'rock'])
##    sns.set_style("whitegrid")
##    indices = np.where( (trainY==c) | (trainY==d))
##    ax2 = sns.jointplot(x=trainX[indices,a], y=trainX[indices,b], kind="kde")
##    ax2.set_axis_labels(gestures.feature_names[a], gestures.feature_names[b])
##    plt.tight_layout()
    
    # show boxplot for a single feature
    a = 1
    plt.figure()
    ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:,a])
    ax3.set_title(gestures.feature_names[a])
    ax3.set_ylabel(gestures.feature_names[a])
    plt.tight_layout()

    # show feature correlation heatmap
    plt.figure()
    corr = np.corrcoef(trainX, rowvar=False)
    ax4 = sns.heatmap(corr, annot=True, xticklabels=gestures.feature_names, yticklabels=gestures.feature_names)
    plt.tight_layout()
    
    
    plt.show(block=True)

