#include "cache.h"

using namespace std;


Cache::Cache(Depth maxdepth, WipeType wipe_type, Size maxcachesize, bool depthAgnostic): wipe_type(wipe_type), maxdepth(maxdepth), maxcachesize(maxcachesize), depthAgnostic(depthAgnostic) {
    cachesize = 0;
}



