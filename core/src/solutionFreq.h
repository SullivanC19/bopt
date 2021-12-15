//
// Created by Gael Aglin on 17/04/2021.
//

#ifndef SOLUTION_FREQ_H
#define SOLUTION_FREQ_H

#include "solution.h"
#include "nodeDataManagerFreq.h"

/**
 * This structure a decision tree model learnt from input data
 * @param expression - a json string representing the tree
 * @param size - the number of nodes (branches + leaves) in the tree
 * @param depth - the depth of the tree; the length of the longest rule in the tree
 * @param trainingError - the error of the tree on the training set given the used objective function
 * @param latSize - the number of nodes explored before finding the solution. Currently this value is not correct :-(
 * @param searchRt - the time that the search took
 * @param timeout - a boolean variable to represent the fact that the search reached a timeout or not
 */
struct Freq_Tree : Tree {

    string to_str() const {
        string out = "";
        out += "Tree: " + expression + "\n";
        out += (expression != "(No such tree)") ? "Size: " + to_string(size) + "\n" : "Size: 0\n";
        out += (expression != "(No such tree)") ? "Depth: " + to_string(depth) + "\n" : "Depth: 0\n";
        out += (expression != "(No such tree)") ? "Error: " + custom_to_str(trainingError) + "\n" : "Error: inf\n";
        out += (expression != "(No such tree)") ? "Accuracy: " + custom_to_str(accuracy) + "\n" : "Accuracy: 0\n";
        out += "CacheSize: " + to_string(cacheSize) + "\n";
        out += "RunTime: " + custom_to_str(runtime) + "\n";
        if (timeout) out += "Timeout: True\n";
        else out += "Timeout: False\n";
        return out;
        /*if (expression != "(No such tree)") {
            out += "Size: " + to_string(size) + "\n";
            out += "Depth: " + to_string(depth) + "\n";
            out += "Error: " + to_string(trainingError) + "\n";
            out += "Accuracy: " + to_string(accuracy) + "\n";
        }*/
    }
};


class SolutionFreq : public Solution {
public:
    SolutionFreq(void*, NodeDataManager*);

    ~SolutionFreq();

    Tree * getTree();

//    virtual void printTimeOut(Tree* tree );
    void printResult(Freq_NodeData *data);

//    virtual Error getTrainingError(const string &tree_json) {}

    Freq_Tree* tree;

protected:
    int printResult(Freq_NodeData *node_data, int depth);
};



#endif //SOLUTION_FREQ_H