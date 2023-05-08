//
// Created by Gael Aglin on 19/10/2021.
//

#include "search_base.h"

Search_base::Search_base(
        NodeDataManager *nodeDataManager,
        bool infoGain,
        bool infoAsc,
        bool repeatSort,
        Support minsup,
        Depth maxdepth,
        int timeLimit,
        Cache *cache,
        float maxError,
        bool specialAlgo,
        bool stopAfterError,
        bool from_cpp,
        int k,
        function<float(int)> *split_penalty_callback_pointer
) :
        nodeDataManager(nodeDataManager),
        infoGain(infoGain),
        infoAsc(infoAsc),
        repeatSort(repeatSort),
        minsup(minsup),
        maxdepth(maxdepth),
        timeLimit(timeLimit),
        cache(cache),
        maxError(maxError),
        specialAlgo(specialAlgo),
        stopAfterError(stopAfterError),
        from_cpp(from_cpp),
        k(k),
        split_penalty_callback_pointer(split_penalty_callback_pointer) {}
