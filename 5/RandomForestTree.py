import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import model_selection
from sklearn.model_selection import train_test_split


def get_bootstrap(data, labels, n_trees):
    # размер совпадает с исходной выборкой
    n_samples = data.shape[0]
    bootstrap = []

    for i in range(n_trees):
        sample_index = np.random.randint(0, n_samples, size=n_samples)
        b_data = data[sample_index]
        b_labels = labels[sample_index]
        bootstrap.append((b_data, b_labels))

    return bootstrap


# Реализуем класс узла
class Node:
    def __init__(self, index, t, true_branch, false_branch):
        # индекс признака, по которому ведется сравнение с порогом в этом узле
        self.index = index
        # значение порога
        self.t = t
        # поддерево, удовлетворяющее условию в узле
        self.true_branch = true_branch
        # поддерево, не удовлетворяющее условию в узле
        self.false_branch = false_branch


class Leaf:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = self.predict()

    def predict(self):
        # подсчет количества объектов разных классов
        # сформируем словарь "класс: количество объектов"
        classes = {}
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1

        # найдем класс, количество объектов которого будет максимальным в этом листе и вернем его
        prediction = max(classes, key=classes.get)
        return prediction


class MyTree:
    def __init__(self):
        self.tree = None

    # Расчет критерия Джини
    @staticmethod
    def gini(labels):
        #  подсчет количества объектов разных классов
        classes = {}
        for label in labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1

        #  расчет критерия
        impurity = 1
        for label in classes:
            p = classes[label] / len(labels)
            impurity -= p ** 2

        return impurity

    # Расчет прироста
    def gain(self, left_labels, right_labels, root_gini):
        # доля выборки, ушедшая в левое поддерево
        p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])
        return root_gini - p * self.gini(left_labels) - (1 - p) * self.gini(right_labels)

    # Разбиение датасета в узле
    @staticmethod
    def split(data, labels, column_index, t):
        left = np.where(data[:, column_index] <= t)
        right = np.where(data[:, column_index] > t)

        true_data = data[left]
        false_data = data[right]

        true_labels = labels[left]
        false_labels = labels[right]

        return true_data, false_data, true_labels, false_labels

    @staticmethod
    def get_subsample(len_sample):
        # будем сохранять не сами признаки, а их индексы
        sample_indexes = list(range(len_sample))
        len_subsample = int(np.sqrt(len_sample))
        subsample = np.random.choice(sample_indexes, size=len_subsample, replace=False)

        return subsample

    # Нахождение наилучшего разбиения
    def find_best_split(self, data, labels):

        #  обозначим минимальное количество объектов в узле
        min_leaf_samples = 5

        root_gini = self.gini(labels)

        best_gain = 0
        best_t = None
        best_index = None

        n_features = data.shape[1]

        # выбираем случайные признаки
        feature_subsample_indices = self.get_subsample(n_features)

        for index in feature_subsample_indices:
            # будем проверять только уникальные значения признака, исключая повторения
            t_values = np.unique(data[:, index])

            for t in t_values:
                true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)
                #  пропускаем разбиения, в которых в узле остается менее 5 объектов
                if len(true_data) < min_leaf_samples or len(false_data) < min_leaf_samples:
                    continue

                current_gain = self.gain(true_labels, false_labels, root_gini)

                #  выбираем порог, на котором получается максимальный прирост качества
                if current_gain > best_gain:
                    best_gain, best_t, best_index = current_gain, t, index

        return best_gain, best_t, best_index

    # Построение дерева с помощью рекурсивной функции

    def build_tree(self, data, labels):

        gain, t, index = self.find_best_split(data, labels)

        #  Базовый случай - прекращаем рекурсию, когда нет прироста в качества
        if gain == 0:
            return Leaf(data, labels)

        true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)

        # Рекурсивно строим два поддерева
        true_branch = self.build_tree(true_data, true_labels)
        false_branch = self.build_tree(false_data, false_labels)

        # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
        return Node(index, t, true_branch, false_branch)

    # Функция классификации отдельного объекта
    def classify_object(self, obj, node):

        #  Останавливаем рекурсию, если достигли листа
        if isinstance(node, Leaf):
            answer = node.prediction
            return answer

        if obj[node.index] <= node.t:
            return self.classify_object(obj, node.true_branch)
        else:
            return self.classify_object(obj, node.false_branch)

    # функция формирования предсказания по выборке на одном дереве
    def predict(self, data, tree):
        classes = []
        for obj in data:
            prediction = self.classify_object(obj, tree)
            classes.append(prediction)
        return classes

    def fit(self, data, labels):
        self.tree = self.build_tree(data, labels)
        return self


class Tree:
    def __init__(self,
                 max_tree_depth_stop=np.inf,
                 max_leaf_num_stop=np.inf,
                 min_leaf_object_stop=1):
        self.max_depth = max_tree_depth_stop
        self.nodes = []
        self.leaves = []
        self.depth = 0
        self.max_leaves = max_leaf_num_stop
        self.min_objects = min_leaf_object_stop
        self.tree = None

    # Расчет критерия Джини
    def gini(self,
             labels):
        #  подсчет количества объектов разных классов
        classes = {}
        for label in labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1

        #  расчет критерия
        impurity = 1
        for label in classes:
            p = classes[label] / len(labels)
            impurity -= p ** 2

        return impurity

    # Расчет прироста
    def gain(self,
             left_labels,
             right_labels,
             root_gini):

        # доля выборки, ушедшая в левое поддерево
        p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])

        return root_gini - p * self.gini(left_labels) - (1 - p) * self.gini(right_labels)

    # Разбиение датасета в узле
    def split(self,
              data,
              labels,
              column_index,
              t):

        left = np.where(data[:, column_index] <= t)
        right = np.where(data[:, column_index] > t)

        true_data = data[left]
        false_data = data[right]

        true_labels = labels[left]
        false_labels = labels[right]

        return true_data, false_data, true_labels, false_labels

    # Нахождение наилучшего разбиения
    def find_best_split(self,
                        data,
                        labels):

        #  обозначим минимальное количество объектов в узле
        min_samples_leaf = 5

        root_gini = self.gini(labels)

        best_gain = 0
        best_t = None
        best_index = None

        n_features = data.shape[1]

        for index in range(n_features):
            # будем проверять только уникальные значения признака, исключая повторения
            t_values = np.unique(data[:, index])

            for t in t_values:
                true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)
                #  пропускаем разбиения, в которых в узле остается менее 5 объектов
                if len(true_data) < min_samples_leaf or len(false_data) < min_samples_leaf:
                    continue

                current_gain = self.gain(true_labels, false_labels, root_gini)

                #  выбираем порог, на котором получается максимальный прирост качества
                if current_gain > best_gain:
                    best_gain, best_t, best_index = current_gain, t, index

        return best_gain, best_t, best_index

    # Построение дерева с помощью рекурсивной функции
    def build_tree(self,
                   data,
                   labels):

        gain, t, index = self.find_best_split(data, labels)

        # ИЗМЕНЕНИЯ: здесь добавила базовые случаи для остановки построения дерева

        #  Базовый случай 2 - прекращаем рекурсию, когда достигли максимальной глубины дерева
        if self.depth > self.max_depth:
            self.leaves.append(Leaf(data, labels))
            return Leaf(data, labels)

        #  Базовый случай 3 - прекращаем рекурсию, когда достигли максимального количества листьев
        if len(self.leaves) >= self.max_leaves - 1 or self.depth >= self.max_leaves - 1:
            self.leaves.append(Leaf(data, labels))
            return Leaf(data, labels)

        #  Базовый случай 4 - прекращаем рекурсию, когда достигли минимального количества объектов в листе
        if len(data) <= self.min_objects:
            self.leaves.append(Leaf(data, labels))
            return Leaf(data, labels)

        #  Базовый случай 1 - прекращаем рекурсию, когда нет прироста в качества
        if gain == 0:
            self.leaves.append(Leaf(data, labels))
            return Leaf(data, labels)

        self.depth += 1

        true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)

        # Рекурсивно строим два поддерева
        true_branch = self.build_tree(true_data, true_labels)
        false_branch = self.build_tree(false_data, false_labels)

        # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
        self.nodes.append(Node(index, t, true_branch, false_branch))
        return Node(index, t, true_branch, false_branch)

    def classify_object(self,
                        obj,
                        node):

        #  Останавливаем рекурсию, если достигли листа
        if isinstance(node, Leaf):
            answer = node.prediction
            return answer

        if obj[node.index] <= node.t:
            return self.classify_object(obj, node.true_branch)
        else:
            return self.classify_object(obj, node.false_branch)

    def fit(self, data, labels):
        self.tree = self.build_tree(data, labels)
        return self

    def predict(self, data):
        classes = []
        for obj in data:
            prediction = self.classify_object(obj, self.tree)
            classes.append(prediction)
        return classes


class RandomForest:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.forest = None

    def fit(self, data, labels):
        self.forest = []
        bootstrap = get_bootstrap(data, labels, self.n_trees)

        for b_data, b_labels in bootstrap:
            tree = Tree()
            tree.fit(b_data, b_labels)
            self.forest.append(tree)
        return self.forest

    def tree_vote(self, data):
        # добавим предсказания всех деревьев в список
        predictions = []
        for tree in self.forest:
            predictions.append(tree.predict(data))

        # сформируем список с предсказаниями для каждого объекта
        predictions_per_object = list(zip(*predictions))

        # выберем в качестве итогового предсказания для каждого объекта то,
        # за которое проголосовало большинство деревьев
        voted_predictions = []
        for obj in predictions_per_object:
            voted_predictions.append(max(set(obj), key=obj.count))
        return voted_predictions
