import json
import numpy as np
class Node():
    def __init__(self, name, state, reward = 0, parent = None, 
                 q_value_after_state = None, parent_action = None, best_q_value = None,
                decom_q_value_after_state = None, decom_best_q_value = None):
        
        self.parent = parent
        self.q_value_after_state = q_value_after_state
        self.best_q_value = best_q_value
        self.decom_q_value_after_state = decom_q_value_after_state
        self.decom_best_q_value = decom_best_q_value
        
        self.children = []
        self.action_dict = {}
        self.actions = []
        self.state = state
        self.name = name
        self.best_child = None
        self.best_action = None
        self.parent_action = parent_action
        

        
    def add_child(self, sub_node, action = None):
        self.children.append(sub_node)
        self.action_dict[str(action)] = sub_node
        
    def save(self):
        pass
    
    def sort_children(self):
        pass
    
    def print_tree(self, p_state = False, p_after_q_value = False, p_best_q_value = False, p_action = False, tab = 0):
        print('    ' * tab, end = '')
        print(self.name, end = ',')
        if p_state:
            print(self.state, end = ',')
        if p_best_q_value:# and self.best_q_value != float("-inf"):
            print(" best_q:", self.best_q_value, end = ',')
        if p_after_q_value:# and self.q_value_after_state != float("-inf"):
            print(" a_s_q_value:", self.q_value_after_state, end = ',')
        if p_action:
            print(" action:", self.parent_action, end = ',')
        print()
        for child in self.children:
            child.print_tree(p_state = p_state, p_after_q_value = p_after_q_value, 
                             p_best_q_value = p_best_q_value, p_action = p_action, tab = tab + 1)
    
    def print_children_prob(self, node):
        np.set_printoptions(precision = 5)
        output = np.zeros(len(node.children))
        for i, child in enumerate(node.children):
            output[i] = child.best_q_value
#         for child in node.children:
#             print(child.decom_best_q_value)
        print(output)
    def state_process(self):
        separate_state = {
                         "mineral" : self.state[0],
                         "Pylons" : self.state[7],
                         "BOT Marines" : self.state[4],
                         "TOP Marines" : self.state[1],
                         "BOT Banelings" : self.state[5],
                         "TOP Banelings" : self.state[2],
                         "BOT Immortals" : self.state[6],
                         "TOP Immortals" : self.state[3],
                         "state": self.state 
                        }
        					
#         return {"state": self.state}
        return separate_state
    
    def action_process(self):
        if self.parent_action is None:
            action_str = None
        else:
            action_str = self.parent_action
#             for a in self.parent_action:
#                 action_str += str(int(a))
        return {"action" : action_str}
    
    def tree_dict(self, best_child = None, new_name = None, tree_number = "", is_partial = False, is_expand = True):
        if new_name is not None:
            self.name = new_name
        t_dict = {}
#         assert best_child is not None, print(self.name)
        self.tree_num = tree_number
#         print(type(self.state_process()), type(self.best_q_value), type(self.q_value_after_state), type(self.parent_action),)
        t_dict["name"] = self.name
        t_dict.update(self.state_process())
        t_dict.update(self.action_process())
#         if self.decom_best_q_value[2] + self.decom_best_q_value[3] + self.decom_best_q_value[6] + self.decom_best_q_value[7] != self.best_q_value:
            
#             print(self.decom_best_q_value)
#             print(self.decom_best_q_value[2] + self.decom_best_q_value[3] + self.decom_best_q_value[6] + self.decom_best_q_value[7])
#             print(self.best_q_value)
#             input()
#         if self.decom_q_value_after_state is not None and self.decom_q_value_after_state[2] + self.decom_q_value_after_state[3] + self.decom_q_value_after_state[6] + self.decom_q_value_after_state[7] != self.q_value_after_state:
#             print("after")
#             print(self.decom_q_value_after_state)
#             print(self.decom_q_value_after_state[2] + self.decom_q_value_after_state[3] + self.decom_q_value_after_state[6] + self.decom_q_value_after_state[7])
#             print(self.q_value_after_state)
#             input()
        t_dict["best q_value"] = self.best_q_value
        t_dict["decom best q_value"] = self.decom_best_q_value
        t_dict["after state q_value"] = self.q_value_after_state
        t_dict["decom after state q_value"] = self.decom_q_value_after_state
        t_dict["tree path"] = self.tree_num
#         t_dict["best_child"] = str(self.best_child)
        j = 0
        t_dict["children"] = [[]]
        
        if is_partial and is_expand:
            for i, child in enumerate(self.children):
                sub_tree_number = tree_number + str(i)
                if child == best_child:
                    new_name = "{}".format(child.name)
                    c_t_dict = child.tree_dict(best_child = child.best_child, new_name = new_name, tree_number = sub_tree_number,
                                              is_partial = is_partial, is_expand = True)
                else:
                    new_name = "{}".format(child.name)
                    c_t_dict = child.tree_dict(new_name = new_name, tree_number = sub_tree_number,
                                              is_partial = is_partial, is_expand = False)
                t_dict["children"][0].append(c_t_dict)
        elif not is_partial:
            for i, child in enumerate(self.children):
                sub_tree_number = tree_number + str(i)
                if child == best_child:
                    new_name = "{}_{}_best".format(child.name, sub_tree_number)
                    c_t_dict = child.tree_dict(best_child = child.best_child, new_name = new_name, tree_number = sub_tree_number,
                                              is_partial = False, is_expand = False)
                else:
                    new_name = "{}_{}".format(child.name, sub_tree_number)
                    c_t_dict = child.tree_dict(new_name = new_name, tree_number = sub_tree_number,
                                              is_partial = False, is_expand = False)
                t_dict["children"][0].append(c_t_dict)
        return t_dict
        
    def save_into_json(self, path = "", dp = 0, is_partial = False):
        t_dict = self.tree_dict(self.best_child, is_partial = is_partial)
        t_json = json.dumps(t_dict, indent=4)
#         print(t_json)
        if is_partial:
            with open(path + "partial_decision_point_" + str(dp) + ".json", 'w') as outfile:
                outfile.write(t_json)
        else:
            with open(path + "whole_decision_point_" + str(dp) + ".json", 'w') as outfile:
                outfile.write(t_json)
            
        
    # Give the node find the worst reward sibling
    # add child (sort by reward)
    # get best child
    # get worst child
    # save tree 
    # action dictionary corresponding to [action1][action2]