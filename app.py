from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

matplotlib.use('Agg') 

app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method == 'POST':
            max_depth =  int(request.form['max_depth'])
            criterion = request.form['criterion']
            splitter =  request.form['splitter']
            min_samples_split = int(request.form['min_samples_split'])
            min_samples_leaf = int(request.form['min_samples_leaf'])
            max_features = int(request.form['max_features'])
            max_leaf_nodes = int(request.form['max_leaf_nodes'])
            min_impurity_decrease = float(request.form['min_impurity_decrease'])
            data = pd.read_csv('Social_Network_Ads.csv')
            x = data.iloc[:,2:4].values
            y = data.iloc[:,-1].values
            
            clf = DecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease,max_leaf_nodes=max_leaf_nodes,max_features=max_features, min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,max_depth=max_depth,criterion=criterion,splitter=splitter)
            clf.fit(x,y)
            
            a = np.arange(start=x[:,0].min()-1,stop=x[:,0].max()+1,step=0.1)
            b = np.arange(start=x[:,1].min()-1,stop=x[:,1].max()+1,step=100)
            
            XX, YY = np.meshgrid(a,b)
            predicted_labels = clf.predict(x)
            y_flattened = y.ravel() if len(y.shape) > 1 else y
            accuracy = accuracy_score(y_flattened, predicted_labels)
        
            
            
            input_array = np.array([XX.ravel(),YY.ravel()]).T
            

            labels = clf.predict(input_array)
            
            plt.figure(figsize=(35,35))
            plot_tree(clf)
            plot_path1 = f'static/images1/plot_{int(time.time())}.png'  # Append random timestamp as query parameter
            plt.savefig(plot_path1)
            plt.figure(figsize=(6,6))
            plt.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.5)
            plt.scatter(x[:,0],x[:,1],c=y)
            plot_path = f'static/images/plot_{int(time.time())}.png'  # Append random timestamp as query parameter
            plt.savefig(plot_path)
            
            return render_template('index.html', plot_path=plot_path,plot_path1=plot_path1,accuracy=accuracy)
        
    return render_template('index.html')




if __name__ =='__main__':
    app.run(debug=True)
