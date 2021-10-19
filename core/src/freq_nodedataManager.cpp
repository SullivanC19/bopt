#include "freq_nodedataManager.h"
#include "cache.h"
#include <iostream>
#include <stack>
#include "logger.h"

Freq_NodeDataManager::Freq_NodeDataManager(
        RCover* cover,
                                 function<vector<float>(RCover *)> *tids_error_class_callback,
                                 function<vector<float>(RCover *)> *supports_error_class_callback,
                                 function<float(RCover *)> *tids_error_callback
                                 ) :
        NodeDataManager(
                cover,
        tids_error_class_callback,
        supports_error_class_callback,
        tids_error_callback)
        /*Query_Best(minsup,
                   maxdepth,
                   trie,
                   data,
                   experror,
                   timeLimit,
                   continuous,
                   tids_error_class_callback,
                   supports_error_class_callback,
                   tids_error_callback,
                   (maxError <= 0) ? NO_ERR : maxError,
                   (maxError <= 0) ? false : stopAfterError) */
                   {}


Freq_NodeDataManager::~Freq_NodeDataManager() {}


//bool Frequency_NodeDataManager::is_freq(pair<Supports, Support> supports) {
//    return supports.second >= minsup;
//}
//
//bool Frequency_NodeDataManager::is_pure(pair<Supports, Support> supports) {
//    Support majnum = supports.first[0], secmajnum = 0;
//    for (int i = 1; i < nclasses; ++i)
//        if (supports.first[i] > majnum) {
//            secmajnum = majnum;
//            majnum = supports.first[i];
//        } else if (supports.first[i] > secmajnum)
//            secmajnum = supports.first[i];
//    return ((long int) minsup - (long int) (supports.second - majnum)) > (long int) secmajnum;
//}

void removeNotBest(Array<Item> itemset, Attribute toDel, Cache* cache){
    if (toDel == -1) return;

    Array<Item> leftI = addItem(itemset, item(toDel, 0));
    Attribute left_attr = (((Freq_NodeData *)cache->get(leftI)->data)->left) ? ((Freq_NodeData *)cache->get(leftI)->data)->test : -1;
    Array<Item> rightI = addItem(itemset, item(toDel, 1));
    Attribute right_attr = (((Freq_NodeData *)cache->get(rightI)->data)->left) ? ((Freq_NodeData *)cache->get(rightI)->data)->test : -1;

    removeNotBest(leftI, left_attr, cache);
    cache->removeItemset(leftI);
    leftI.free();

    removeNotBest(rightI, right_attr, cache);
    cache->removeItemset(rightI);
    rightI.free();
}

bool Freq_NodeDataManager::updateData(NodeData *best, Error upperBound, Attribute attribute, NodeData *left, NodeData *right, Array<Item> itemset, Cache* cache) {
    auto *freq_best = (Freq_NodeData *) best, *freq_left = (Freq_NodeData *) left, *freq_right = (Freq_NodeData *) right;
    Error error = freq_left->error + freq_right->error;
    Size size = freq_left->size + freq_right->size + 1;
    if (error < upperBound || (floatEqual(error, upperBound) && size < freq_best->size)) {
        if (not cache->with_cache)  {
            cout << "remove subtree with update. ub = " << upperBound << " err foud = " << error << endl;
            cache->removeSubTree(itemset, !(freq_best)->left ? -1 : freq_best->test);
        }
        freq_best->error = error;
        freq_best->left = freq_left;
        freq_best->right = freq_right;
        freq_best->size = size;
        freq_best->test = attribute;
        return true;
    }
    else if (not cache->with_cache) {
        cout << "remove subtree without update. ub = " << upperBound << " err foud = " << error << endl;
        cache->removeSubTree(itemset, attribute);
    }
    return false;
}

NodeData *Freq_NodeDataManager::initData(RCover *cov, Depth currentMaxDepth, int hashcode) {
    Class maxclass = -1;
    Error error;

    auto *data = new Freq_NodeData();

    if (cov == nullptr) cov = cover;

//    if (trie->use_priority) trie->nodemapper.push({cover->getSupport(), hashcode});

    //fast or default error. support will be used
    if (tids_error_class_callback == nullptr && tids_error_callback == nullptr) {
        //python fast error
        if (supports_error_class_callback != nullptr) {
            function<vector<float>(RCover *)> callback = *supports_error_class_callback;
            cov->getSupportPerClass(); // allocate the sup_array if it does not exist yet and compute the frequency counts
            vector<float> infos = callback(cover);
            error = infos[0];
            maxclass = int(infos[1]);
        }
        //default error
        else {
            LeafInfo ev = computeLeafInfo();
            error = ev.error;
            maxclass = ev.maxclass;
        }
    }
    //slow error or predictor error function. Not need to compute support
    else {
        if (tids_error_callback != nullptr) {
            function<float(RCover *)> callback = *tids_error_callback;
            error = callback(cov);
        } else {
            function<vector<float>(RCover *)> callback = *tids_error_class_callback;
            vector<float> infos = callback(cov);
            error = infos[0];
            maxclass = int(infos[1]);
        }
    }
    data->test = maxclass;
    data->leafError = error;
//    data->error += experror->addError(cover->getSupport(), data->error, dm->getNTransactions());
    data->solutionDepth = currentMaxDepth;

    return (NodeData *) data;
}

LeafInfo Freq_NodeDataManager::computeLeafInfo(RCover *cov) {
    if (cov == nullptr) cov = cover;
    Class maxclass;
    Error error;
    Supports itemsetSupport = cov->getSupportPerClass();
    SupportClass maxclassval = itemsetSupport[0];
    maxclass = 0;

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (floatEqual(itemsetSupport[i], maxclassval)) {
            if (cov->dm->getSupports()[i] > cov->dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    error = sumSupports(itemsetSupport) - maxclassval;
    return {error, maxclass};
}


LeafInfo Freq_NodeDataManager::computeLeafInfo(Supports itemsetSupport) {
    Class maxclass = 0;
    Error error;
    SupportClass maxclassval = itemsetSupport[0];

    for (int i = 1; i < nclasses; ++i) {
        if (itemsetSupport[i] > maxclassval) {
            maxclassval = itemsetSupport[i];
            maxclass = i;
        } else if (floatEqual(itemsetSupport[i], maxclassval)) {
            if (cover->dm->getSupports()[i] > cover->dm->getSupports()[maxclass])
                maxclass = i;
        }
    }
    error = sumSupports(itemsetSupport) - maxclassval;
    return {error, maxclass};
}
