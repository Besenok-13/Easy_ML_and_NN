from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class DBScan:
    def __init__(self, num_neib, max_dist):
        self.num_neib = num_neib
        self.max_dist = max_dist
        self.groups = []
        self.red_flags = set()

        #Для не наивной реализации
        self.karandash = dict()
    
    
    def data_to_numpy(self, data:pd.DataFrame):
        return np.array(data, dtype=np.float64)


    def data_save_inds(self, data:np.array):
        return np.array([[i, point] for i, point in enumerate(data)], dtype=list)
    

    def del_point(self, points_to_del_ind:int):
        points_to_del_ind = np.array([points_to_del_ind]).reshape(-1)
        self.points_indexs_queue  = np.fromiter((point_ind for point_ind in self.points_indexs_queue if point_ind not in points_to_del_ind), dtype=int)


    def get_random_point(self, points_inds:np.array):
        now_ind = np.random.choice(points_inds)
        return (self.points_queue[now_ind][0], self.points_queue[now_ind][1])


    def search_neibs(self, now_point:np.array, points:np.array):
        # np.array([[point_ind, np.sqrt(np.sum((now_point - point)**2)), point] for point_ind, point in points if np.sqrt(np.sum((now_point - point)**2)) < self.max_dist], dtype=list)
        return [[point_ind, point] for point_ind, point in points if np.sqrt(np.sum((now_point - point)**2)) < self.max_dist]



    def check_group(self, neibs:list):
        #TODO Почему-то разбивает точки одного кластера на разные
        #Потому, что нампаевское объединение не прибавляет к существующему массиву, а создаёт новый
        for point_ind, neib in neibs:
            self.del_point(points_to_del_ind=point_ind)
            #посмотреть что можно сделать с point_ind и таким образлм связать его с жёлтыми индексами
            
            #добавляем соседа в любом случае, если он контачит с зелёным, не важно, жёлтый он или нет. Вот это нужно будет переписать
            #в случае создания логики основанной на расстоянии до ближайшего зелёного
            self.groups[-1] += [point_ind]
            # now_neibs = self.search_neibs(now_point=neib, points=self.points_queue[self.points_indexs_queue])
            now_neibs = self.search_neibs(now_point=neib, points=self.points_queue)
            #Если и эта точка является зелёной, то добавляем её соседей в список, который необходимо обойти
            if len(now_neibs) >= self.num_neib:
                
                #находим соседей, которые ещё не находятся в очереди на обработку
                soseds_to_add = [sosed for sosed in  now_neibs if ((sosed[0] not in np.array(neibs, dtype=list)[:,0] ) and (sosed[0] in self.points_indexs_queue))]

                if len(soseds_to_add):
                    neibs += soseds_to_add
                
            #если уже находили эту точку ранее и она оказалась в жёлтых - удаляем её оттуда.
            else:
                self.karandash.pop(str(point_ind), None)
    def calc_dists(self, centroids:np.array, data:np.array):
        return np.array([[[np.sqrt(np.sum((centroid - point)**2)), i] for centroid in centroids] for i, point in enumerate(data)], dtype=np.float64)

    def fit(self, points:pd.DataFrame):
        self.points = self.data_to_numpy(data=points)
        self.points_queue = self.data_to_numpy(data=points)
        self.points_indexs_queue = np.arange(self.points_queue.shape[0])
        self.points_queue =self.data_save_inds(data=self.points_queue)

        while len(self.points_indexs_queue):
            
            #вот тут приколы с бесконечным выбором одного и той же жёлтой точки
            point_ind, now_point = self.get_random_point(self.points_indexs_queue)
            self.del_point(points_to_del_ind=point_ind)


            # print(point_ind, now_point)
            # print(self.points_indexs_queue)
            # neibs = self.search_neibs(now_point=now_point, points = self.points_queue[self.points_indexs_queue])
            # Реально сводит количество наблюдений к нормальному числу self.points_queue[self.points_indexs_queue]
            neibs = self.search_neibs(now_point=now_point, points = self.points_queue[self.points_indexs_queue])
            if len(neibs) < self.num_neib:
                #наивная реализация. В случае отношения точки к жёлтым флагам, она присвоится к тому кластеру, который найдёт её первым
                if len(neibs)==0:
                    self.red_flags.update([point_ind])
                else:
                    self.karandash.update({str(point_ind): now_point})

                #Скорее всего здесь не обрабатываются точки рядом с которыми что-то есть, но их меньще чем self.num_neib
            else:
                self.groups += [[point_ind]]
                self.check_group(neibs=neibs)
        
        self.red_flags.update(list(self.karandash.keys()))
        self.red_flags = [int(i) for i in self.red_flags]


    def drow(self):
        for group in self.groups:
            print("hahahahahaaha")
            print(self.points[group])
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            plt.scatter(self.points[group][:,0], self.points[group][:,1])
        
        print("RED FLAAAAAAG", self.red_flags)
        plt.scatter(self.points[self.red_flags][:,0], self.points[self.red_flags][:,1])
        plt.show()


iris = datasets.load_iris()
# dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
# dataset
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(iris.shape)
# iris.plot(kind="scatter", x='sepal length (cm)', y='sepal width (cm)')

# sns.jointplot(x='sepal length (cm)', y='sepal width (cm)', data=iris, size=5)
iris.head(3)
sns.FacetGrid(iris, hue="target", height=5) \
   .map(plt.scatter, 'sepal length (cm)', 'sepal width (cm)') \
   .add_legend()
plt.show()

a = DBScan(num_neib=9, max_dist=0.3)
a.fit(points=iris[['sepal length (cm)', 'sepal width (cm)']])
a.drow()

sy=0
for i in a.groups:
    sy += len(i)

print(sy + len(a.red_flags))