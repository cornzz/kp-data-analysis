import numpy as np
import pandas as pd


class Node:
    def __init__(self, value, children, branch_value=None):
        """
        :param value: split attribute or, if leaf node, class label
        :param children: array of child nodes
        :param branch_value: value of branch coming from parent node
        """
        self.value = value
        self.children = children
        self.branch_value = branch_value

    # Copied from first stackoverflow result when searching 'print tree python' and modified
    # for a quick and dirty printable representation of the decision tree
    def __repr__(self, level=0):
        ret = "\t" * level + (self.branch_value + ': ' if self.branch_value is not None else '') + self.value + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret


def entropy(examples):
    """
    :param examples: list of probabilities
    :return: resulting entropy
    """
    p = examples[class_index].value_counts() / len(examples)
    return np.sum([-px * np.log2(px) for px in p])


def information_gain(examples, target_attr):
    """
    :param examples: set of examples to be split
    :param target_attr: attribute to be used for split
    :return: information gain achieved if split was performed
    """
    unique_attributes = examples[target_attr].unique()
    remainder = 0
    for u_attr in unique_attributes:
        u_attr_examples = examples[examples[target_attr] == u_attr]
        remainder += len(u_attr_examples) / len(examples) * entropy(u_attr_examples)
    return entropy(examples) - remainder


def choose_best_attr(examples, attributes):
    """
    :param examples: set of examples for which the best split attribute is to be determined
    :param attributes: list of attributes on which the data can be split
    :return: split attribute that yields the maximum information gain
    """
    ig = {attr: information_gain(examples, attr) for attr in attributes}
    return max(ig, key=ig.get)


def gen_tree(examples, attributes, branch_value=None):
    """
    :param examples: set of examples which are used for the recursive splitting
    :param attributes: list of attributes on which the data can be split
    :param branch_value: value of branch coming from parent node
    :return: decision tree for given arguments
    """
    if examples[class_index].nunique() == 1:
        # All examples have same class label, return leaf node with class label
        return Node(examples.iloc[0, -1], [], branch_value)
    elif len(attributes) == 0:
        # No attributes left to split on, return leaf node with majority class label
        return Node(examples[class_index].mode()[0], [], branch_value)
    else:
        # Perform recursive split on attribute which yields highest IG
        best = choose_best_attr(examples, attributes)
        children = []
        # Create child node for each possible attribute value
        for av in attribute_values[best]:
            examples_av = examples[examples[best] == av]
            if len(examples_av) == 0:
                # No examples with attribute value left, return leaf node with majority class label of parent node
                children.append(Node(examples[class_index].mode()[0], [], av))
            else:
                # Recursively generate subtree for attribute value av and examples containing av
                children.append(gen_tree(examples_av, [a for a in attributes if a != best], av))
        return Node(best, children, branch_value)


def classify(node, d):
    """ Recursively descends into decision tree according to given decision problem

    :param node: root of decision tree to be used for classification
    :param d: decision problem dict with attributes as keys
    :return: class label
    """
    if len(node.children) == 0:
        return node.value
    attribute_value = d[node.value]
    return classify(next(filter(lambda child: child.branch_value == attribute_value, node.children)), d)


df = pd.read_csv('data-cls.csv')
class_index = 'tennis'
all_attributes = df.columns.values[:-1]
attribute_values = {attr: df[attr].unique() for attr in all_attributes}
tree = gen_tree(df, all_attributes)
decision_problem = {'forecast': 'rainy', 'temperature': 'hot', 'humidity': 'high', 'wind': 'strong'}
print(tree)
print('------------------ Task 1 ------------------')
print(f'Classification of problem {decision_problem}: {classify(tree, decision_problem)}')
