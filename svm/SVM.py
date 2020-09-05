import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

#  style to emulate ggplot of R
style.use('ggplot')


data = {-1:np.array([[1,8],[2,8],[3,8]]),1:np.array([[5,1],[6,-1],[7,3]])}

class Support_Vector_Machine:
    def __init__(self,visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization :
            self.fig = plt.figure()
            # a subplot with 1,1 grids on plot number 1
            self.ax = self.fig.add_subplot(1,1,1)
            
    # train
    def fit(self,data):
        self.data = data
        # {||w|| : [w,b]}
        opt = {}
        # Different Values For Same Magnitude Of 'W'
        transforms =   [[1,1],[-1,1],[1,-1],[-1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # Free Memory
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]

        # cost: Extremely Expensive
        b_range_multiple = 10

        #
        b_multiple = 10

        # just a starting point for w
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum]) 
            # We can do this because Convex !
            optimized = False 
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                            self.max_feature_value*b_range_multiple, 
                            step*b_multiple):  
                    
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally 
                        # SMO attempts to fix this a bit
                        # Yi(Xi.W + b) >= 1
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                # print(yi*(np.dot(w_t,xi)+b))
                                    
                        if found_option:
                            opt[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print("Optimized a step ... ")
                else:
                    # Example : 
                    # w = [5,5]
                    # step = 1
                    # w - [step , step] = [4,4]
                    w = w - step
            norms = sorted([n for n in opt])
            # ||w|| : [w.b]
            opt_choice = opt[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]  
            latest_optimum = opt_choice[0][0] + step*2

        #  Print Final Values

        for yi in self.data:
            for xi in self.data[yi]:
                print(yi*(np.dot(self.w,xi)+self.b))

    def predict(self,features):
        # sign( x.w + b ) 
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)

        if classification != 0 and self.visualization:
            self.ax.scatter(features[0] , features[1],s=200,marker='*',c=self.colors[classification])

        return classification
        

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=200,color=self.colors[i]) for x in data[i]] for i in data]

        # hyper plane : x.w + b   
        # v = x.w + b
        # positive support vector = 1
        # negative support vector = -1
        # decision boundary = 0
        def hyperplane(x,w,b,v):
            # y = x.w + b -> y - x.w - b 
            return (-w[0]*x -b + v) / w[1]

        datarange = (self.min_feature_value * 0.9 , self.max_feature_value*1.1)
        hyp_x_min = datarange[0]        
        hyp_x_max = datarange[1]


        # Positive SV
        psv1 = hyperplane(hyp_x_min,self.w,self.b,1)
        psv2 = hyperplane(hyp_x_max,self.w,self.b,1)

        # k -> black
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')

        # Negative SV
        nsv1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2 = hyperplane(hyp_x_max,self.w,self.b,-1)

        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')

        # Desision Boundary
        db1 = hyperplane(hyp_x_min,self.w,self.b,0)
        db2 = hyperplane(hyp_x_max,self.w,self.b,0)

        # y-- -> dashed yellow
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--') 

        plt.show()


svm = Support_Vector_Machine()

svm.fit(data)

predict_vals = [[0,10],
                [1,3],
                [3,4],
                [3,5],
                [5,5],
                [6,-5],
                [5,8],]

for i in predict_vals:
    svm.predict(i)

svm.visualize()