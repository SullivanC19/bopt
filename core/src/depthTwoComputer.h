//
// Created by Gael Aglin on 26/09/2020.
//
#ifndef DL85_DEPTHTWOCOMPUTER_H
#define DL85_DEPTHTWOCOMPUTER_H

/*#include "rCover.h"
#include "trie.h"
#include "query.h"
#include "query_best.h"
#include <chrono>
#include <utility>*/

#include "rCover.h"
#include "cache.h"
#include "solution.h"
#include "freq_nodedataManager.h"
#include "rCoverTotalFreq.h"
//#include "lcm_pruned.h"
#include <chrono>
#include <utility>

using namespace std::chrono;

class Search_base;
//class Search;
//class Search_nocache;

pair<Node*, Error> computeDepthTwo(RCover*, Error, Array<Attribute>, Attribute, Array<Item>, Node*, NodeDataManager*, Error, Cache*, Search_base*);

struct TreeTwo{
    Freq_NodeData* root_data;

    TreeTwo(){
        root_data = new Freq_NodeData();
    }

    void replaceTree(TreeTwo* cpy){
        free();
        root_data = cpy->root_data;
    }

    void free(){
        if (root_data->left || root_data->right){
            if (root_data->left->left || root_data->left->right){
                delete root_data->left->left;
                delete root_data->left->right;
            }
            if (root_data->right->left || root_data->right->right){
                delete root_data->right->left;
                delete root_data->right->right;
            }
            delete root_data->left;
            delete root_data->right;
        }
        delete root_data;
    }


    ~TreeTwo(){
        free();
    }
};

#endif //DL85_DEPTHTWOCOMPUTER_H