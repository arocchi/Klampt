#ifndef MOTIONPLANNING_H
#define MOTIONPLANNING_H

#include <string>

// Forward declaration of C-type PyObject
struct _object;
typedef _object PyObject;

/// Sets the random seed used by the motion planner
void setRandomSeed(int seed);

/// Loads planner values from a JSON string
void setPlanJSONString(const char* string);
/// Saves planner values to a JSON string
std::string getPlanJSONString();

/** @brief Sets the planner type.
 *
 * Valid values are
 * - prm: the Probabilistic Roadmap algorithm
 * - rrt: the Rapidly Exploring Random Trees algorithm
 * - sbl: the Single-Query Bidirectional Lazy planner
 * - sblprt: the probabilistic roadmap of trees (PRT) algorithm with SBL as the inter-root planner.
 * - rrt*: the RRT* algorithm for optimal motion planning 
 * - prm*: the PRM* algorithm for optimal motion planning
 * - lazyprm*: the Lazy-PRM* algorithm for optimal motion planning
 * - lazyrrg*: the Lazy-RRG* algorithm for optimal motion planning
 * - fmm: the fast marching method algorithm for resolution-complete optimal motion planning
 * - fmm*: an anytime fast marching method algorithm for optimal motion planning
 */
void setPlanType(const char* type);

void setPlanSetting(const char* setting,double value);

/** @brief Sets a numeric or string-valued setting for the planner.
 *
 * Valid numeric values are:
 * - "knn": k value for the k-nearest neighbor connection strategy (only for
 *   PRM)
 * - "connectionThreshold": a milestone connection threshold
 * - "perturbationRadius": (for RRT and SBL)
 * - "bidirectional": 1 if bidirectional planning is requested (for RRT)
 * - "grid": 1 if a point selection grid should be used (for SBL)
 * - "gridResolution": resolution for the grid, if the grid should be used 
 *   (for SBL with grid, FMM, FMM*)
 * - "suboptimalityFactor": allowable suboptimality (for RRT*, lazy PRM*,
 *   lazy RRG*)
 * - "randomizeFrequency": a grid randomization frequency (for SBL)
 * - "shortcut": nonzero if you wish to perform shortcutting after a first
 *   plan is found.
 * - "restart": nonzero if you wish to restart the planner to get better
 *   paths with the remaining time.
 * 
 * Valid string values are:
 * - "pointLocation": a string designating a point location data structure.
 *   "kdtree" is supported, optionally followed by a weight vector (for
 *   PRM, RRT*, PRM*, LazyPRM*, LazyRRG*)
 * - "restartTermCond": used if the "restart" setting is true.  This is a
 *   JSON string defining the termination condition (default value:
 *   "{foundSolution:1;maxIters:1000}")
*/
void setPlanSetting(const char* setting,const char* value);

///Performs cleanup of all created spaces and planners
void destroy();

/** @brief A raw interface for a configuration space.  Note: the native
 * Python CSpace interface class in cspace.py is easier to use.
 *
 * You can either set a single feasibility test function using setFeasibility()
 * or add several feasibility tests, all of which need to be satisfied, using
 * addFeasibilityTest().  In the latter case, planners may be able to provide
 * debugging statistics, solve Minimum Constraint Removal problems, run faster
 * by eliminating constraint tests, etc.
 *
 * Either setVisibility() or setVisibilityEpsilon() must be called to define
 * a visibility checker between two (feasible) configurations.  In the 
 * latter case, the path will be discretized at the resolution sent to
 * setVisibilityEpsilon.  If you have special single-constraint visibility
 * tests, you can call that using addVisibilityTest (for example, for convex
 * constraints you can set it to the lambda function that returns true
 * regardless of its arguments).
 *
 * Supported properties include "euclidean" (boolean), "metric" (string),
 * "geodesic" (boolean), "minimum" (vector), and "maximum" (vector). 
 * These may be used by planners to make planning faster or more accurate.
 * For a complete list see KrisLibrary/planning/CSpace.h.
 */
class CSpaceInterface
{
 public:
  CSpaceInterface();
  CSpaceInterface(const CSpaceInterface&);  //copy constructor
  ~CSpaceInterface();
  void destroy();
  void setFeasibility(PyObject* pyFeas);
  void addFeasibilityTest(const char* name,PyObject* pyFeas);
  void setVisibility(PyObject* pyVisible);
  void addVisibilityTest(const char* name,PyObject* pyVisible);
  void setVisibilityEpsilon(double eps);
  void setSampler(PyObject* pySamp);
  void setNeighborhoodSampler(PyObject* pySamp);
  void setDistance(PyObject* pyDist);
  void setInterpolate(PyObject* pyInterp);
  void setProperty(const char* key,const char* value);
  const char* getProperty(const char* key);

  ///queries
  bool isFeasible(PyObject* q);
  bool isVisible(PyObject* a,PyObject* b);
  bool testFeasibility(const char* name,PyObject* q);
  bool testVisibility(const char* name,PyObject* a,PyObject* b);
  PyObject* feasibilityFailures(PyObject* q);
  PyObject* visibilityFailures(PyObject* a,PyObject* b);
  PyObject* sample();
  double distance(PyObject* a,PyObject* b);
  PyObject* interpolate(PyObject* a,PyObject* b,double u);

  int index;
};

/** @brief An interface for a motion planner.  The MotionPlanner interface
 * in cspace.py is somewhat easier to use.
 *
 * On construction, uses the planner type specified by setPlanType
 * and the settings currently specified by calls to setPlanSetting.
 *
 * Point-to-point planning is enabled by sending two configurations to
 * the setEndpoints method.
 * This is mandatory for RRT and SBL-style planners.  The start and end
 * milestones are given by indices 0 and 1, respectively
 *
 * Point-to-set planning is enabled by sending a *goal test* as
 * the second argument to the setEndpoints method.  It is possible also
 * to send a special goal sampler by providing a *pair of functions* as the
 * second argument consisting of the two functions (goaltest,goalsample).
 * The first in this pair  tests whether a configuration is a goal, and
 * the second returns a sampled configuration in a superset of the goal.
 * Ideally the goal sampler generates as many goals as possible.
 *
 * PRM can be used in either point-to-point or multi-query mode.  In 
 * multi-query mode, you may call addMilestone(q) to add a new milestone.
 * addMilestone() returns the index of that milestone, which can be used
 * in later calls to getPath().
 * 
 * To plan, call planMore(iters) until getPath(0,1) returns non-NULL.
 * The return value is a list of configurations.
 *
 * To get a roadmap (V,E), call getRoadmap().  V is a list of configurations
 * (each configuration is a Python list) and E is a list of edges (each edge is
 * a pair (i,j) indexing into V).
 * 
 * To dump the roadmap to disk, call dump(fn).  This saves to a
 * Trivial Graph Format (TGF) format.
 */
class PlannerInterface
{
 public:
  PlannerInterface(const CSpaceInterface& cspace);
  ~PlannerInterface();
  void destroy();
  bool setEndpoints(PyObject* start,PyObject* goal);
  int addMilestone(PyObject* milestone);
  void planMore(int iterations);
  PyObject* getPathEndpoints();
  PyObject* getPath(int milestone1,int milestone2);
  double getData(const char* setting);
  PyObject* getStats();
  PyObject* getRoadmap();
  void dump(const char* fn);

  int index;
  int spaceIndex;
};

#endif
