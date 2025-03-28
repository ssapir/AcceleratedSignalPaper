/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ProbNMDA
#define _nrn_initial _nrn_initial__ProbNMDA
#define nrn_cur _nrn_cur__ProbNMDA
#define _nrn_current _nrn_current__ProbNMDA
#define nrn_jacob _nrn_jacob__ProbNMDA
#define nrn_state _nrn_state__ProbNMDA
#define _net_receive _net_receive__ProbNMDA 
#define state state__ProbNMDA 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define tau_r _p[0]
#define tau_r_columnindex 0
#define tau_d _p[1]
#define tau_d_columnindex 1
#define Use _p[2]
#define Use_columnindex 2
#define Dep _p[3]
#define Dep_columnindex 3
#define Fac _p[4]
#define Fac_columnindex 4
#define e _p[5]
#define e_columnindex 5
#define mg _p[6]
#define mg_columnindex 6
#define mggate _p[7]
#define mggate_columnindex 7
#define u0 _p[8]
#define u0_columnindex 8
#define i _p[9]
#define i_columnindex 9
#define g _p[10]
#define g_columnindex 10
#define A _p[11]
#define A_columnindex 11
#define B _p[12]
#define B_columnindex 12
#define factor _p[13]
#define factor_columnindex 13
#define DA _p[14]
#define DA_columnindex 14
#define DB _p[15]
#define DB_columnindex 15
#define v _p[16]
#define v_columnindex 16
#define _g _p[17]
#define _g_columnindex 17
#define _tsav _p[18]
#define _tsav_columnindex 18
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 0, 0
};
 /* declare global and static user variables */
#define gmax gmax_ProbNMDA
 double gmax = 0.001;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gmax_ProbNMDA", "us",
 "tau_r", "ms",
 "tau_d", "ms",
 "Use", "1",
 "Dep", "ms",
 "Fac", "ms",
 "e", "mV",
 "mg", "mM",
 "i", "nA",
 "g", "uS",
 0,0
};
 static double A0 = 0;
 static double B0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "gmax_ProbNMDA", &gmax_ProbNMDA,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[2]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ProbNMDA",
 "tau_r",
 "tau_d",
 "Use",
 "Dep",
 "Fac",
 "e",
 "mg",
 "mggate",
 "u0",
 0,
 "i",
 "g",
 0,
 "A",
 "B",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 19, _prop);
 	/*initialize range parameters*/
 	tau_r = 0.29;
 	tau_d = 43;
 	Use = 0.67;
 	Dep = 800;
 	Fac = 3;
 	e = 0;
 	mg = 1;
 	mggate = 0;
 	u0 = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 19;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ProbNMDA_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 19, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 5;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ProbNMDA ProbNMDA.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "NMDA receptor with presynaptic short-term plasticity ";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int state(_threadargsproto_);
 
/*VERBATIM*/
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

float ranfNMDA(){
        //double MAX = (double)RAND_MAX;
        double r = (rand() / (double) RAND_MAX);
        return r;
}

void SetSeedNowNMDA(){
#ifdef SYN_DEBUG
srand(time(NULL));
#else
srand(888);
#endif
return;
}
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DA = - A / tau_r ;
   DB = - B / tau_d ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DA = DA  / (1. - dt*( ( - 1.0 ) / tau_r )) ;
 DB = DB  / (1. - dt*( ( - 1.0 ) / tau_d )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    A = A + (1. - exp(dt*(( - 1.0 ) / tau_r)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau_r ) - A) ;
    B = B + (1. - exp(dt*(( - 1.0 ) / tau_d)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau_d ) - B) ;
   }
  return 0;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   _args[0] = _args[0] * 0.71 ;
   if ( Fac > 0.0 ) {
     _args[3] = _args[3] * exp ( - ( t - _args[4] ) / Fac ) ;
     }
   else {
     _args[3] = Use ;
     }
   if ( Fac > 0.0 ) {
     _args[3] = _args[3] + Use * ( 1.0 - _args[3] ) ;
     }
   _args[1] = 1.0 - ( 1.0 - _args[1] ) * exp ( - ( t - _args[4] ) / Dep ) ;
   _args[2] = _args[3] * _args[1] ;
   _args[1] = _args[1] - _args[3] * _args[1] ;
   _args[4] = t ;
   if ( ranfNMDA ( _threadargs_ ) < _args[2] ) {
       if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = A;
    double __primary = (A + _args[0] * factor) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau_r ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau_r ) - __primary );
    A += __primary;
  } else {
 A = A + _args[0] * factor ;
       }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = B;
    double __primary = (B + _args[0] * factor) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau_d ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau_d ) - __primary );
    B += __primary;
  } else {
 B = B + _args[0] * factor ;
       }
 }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       double* _p = _pnt->_prop->param;
    Datum* _ppvar = _pnt->_prop->dparam;
    Datum* _thread = (Datum*)0;
    NrnThread* _nt = (NrnThread*)_pnt->_vnt;
 _args[1] = 1.0 ;
   _args[3] = u0 ;
   _args[4] = t ;
   }
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  A = A0;
  B = B0;
 {
   double _ltp ;
 A = 0.0 ;
   B = 0.0 ;
   _ltp = ( tau_r * tau_d ) / ( tau_d - tau_r ) * log ( tau_d / tau_r ) ;
   factor = - exp ( - _ltp / tau_r ) + exp ( - _ltp / tau_d ) ;
   factor = 1.0 / factor ;
   SetSeedNowNMDA ( _threadargs_ ) ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   mggate = 1.0 / ( 1.0 + exp ( 0.062 * - ( v ) ) * ( mg / 3.57 ) ) ;
   g = gmax * ( B - A ) * mggate ;
   i = g * ( v - e ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {   state(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = A_columnindex;  _dlist1[0] = DA_columnindex;
 _slist1[1] = B_columnindex;  _dlist1[1] = DB_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "ProbNMDA.mod";
static const char* nmodl_file_text = 
  "TITLE NMDA receptor with presynaptic short-term plasticity \n"
  "\n"
  "\n"
  "COMMENT\n"
  "NMDA receptor conductance using a dual-exponential profile\n"
  "Presynaptic short-term plasticity based on Fuhrmann et al, 2002\n"
  "Implemented by Srikanth Ramaswamy, Blue Brain Project, March 2009\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "    THREADSAFE\n"
  "	POINT_PROCESS ProbNMDA\n"
  "	RANGE tau_r, tau_d\n"
  "	RANGE Use, u, Dep, Fac, u0\n"
  "	RANGE i, g, e\n"
  "	RANGE mg,mggate\n"
  "    NONSPECIFIC_CURRENT i\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	tau_r = 0.29	 	(ms) : dual-exponential conductance profile\n"
  "	tau_d = 43		(ms) : IMPORTANT: tau_r < tau_d\n"
  "	Use = 0.67 		(1)      : Utilization of synaptic efficacy (just initial values! Use,Dep and Fac are overwritten by BlueBuilder assigned values) 	\n"
  "	Dep = 800 		(ms) 	 : relaxation time constant from depression\n"
  "	Fac = 3 		(ms)     :  relaxation time constant from facilitation\n"
  "    e  = 0     (mV)  : NMDA reversal potential	\n"
  "    gmax = 0.001     (us) : weight conversion factor (from nS to uS)\n"
  "	mg = 1			(mM)  : initial concentration of mg2+\n"
  "	mggate       \n"
  "    u0 = 0      :initial value of u, which is the running value of Use\n"
  "}\n"
  "\n"
  "COMMENT\n"
  "The Verbatim block is needed to generate random nos. from a uniform distribution between 0 and 1 \n"
  "for comparison with Pr to decide whether to activate the synapse or not\n"
  "ENDCOMMENT\n"
  "   \n"
  "VERBATIM\n"
  "#include<stdlib.h>\n"
  "#include<stdio.h>\n"
  "#include<math.h>\n"
  "\n"
  "float ranfNMDA(){\n"
  "        //double MAX = (double)RAND_MAX;\n"
  "        double r = (rand() / (double) RAND_MAX);\n"
  "        return r;\n"
  "}\n"
  "\n"
  "void SetSeedNowNMDA(){\n"
  "#ifdef SYN_DEBUG\n"
  "srand(time(NULL));\n"
  "#else\n"
  "srand(888);\n"
  "#endif\n"
  "return;\n"
  "}\n"
  "ENDVERBATIM\n"
  "   \n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "	v (mV)\n"
  "	i (nA)\n"
  "	g (uS)\n"
  "    factor\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	A  : state variable to construct the dual-exponential profile - decays with conductance tau_r\n"
  "	B  : state variable to construct the dual-exponential profile - decays with conductance tau_d\n"
  "}\n"
  "\n"
  "INITIAL{\n"
  "	LOCAL tp\n"
  "	A = 0\n"
  "	B = 0\n"
  "	tp = (tau_r*tau_d)/(tau_d-tau_r)*log(tau_d/tau_r) :time to peak of the conductance\n"
  "	factor = -exp(-tp/tau_r)+exp(-tp/tau_d) :Normalization factor - so that when t = tp, gsyn = gpeak\n"
  "	factor = 1/factor\n"
  "    SetSeedNowNMDA()\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE state METHOD cnexp\n"
  "	mggate = 1 / (1 + exp(0.062 (/mV) * -(v)) * (mg / 3.57 (mM))) :mggate kinetics - Jahr & Stevens 1990\n"
  "	g = gmax*(B - A)*mggate\n"
  "	i = g*(v - e)\n"
  "}\n"
  "\n"
  "DERIVATIVE state {\n"
  "	A' = -A/tau_r\n"
  "	B' = -B/tau_d\n"
  "}\n"
  "\n"
  "NET_RECEIVE (weight, Pv, Pr, u, tsyn (ms)){\n"
  "weight = weight*0.71 \n"
  ":printf(\"weight NMDA = %g \\n \",weight)\n"
  ":the NETCON.weight = gsyn (per synaptic contact) * scaling factor * 0.71, as gNMDA/gAMPA = 0.71 from Chaelon et al. 2003, and Markram et al. 1997\n"
  "    INITIAL{\n"
  "		Pv=1\n"
  "		u=u0\n"
  "		tsyn=t\n"
  "	    }\n"
  "        : calc u at event-\n"
  "    	if (Fac > 0) {\n"
  "	      	u = u*exp(-(t - tsyn)/Fac) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.\n"
  "	   } else {\n"
  "		  u = Use  \n"
  "	   } \n"
  "	   if(Fac > 0){\n"
  "		  u = u + Use*(1-u) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.\n"
  "	   }	\n"
  "\n"
  "        \n"
  "            Pv   = 1 - (1-Pv) * exp(-(t-tsyn)/Dep) :Probability Pv for a vesicle to be available for release, analogous to the pool of synaptic\n"
  "                                                 :resources available for release in the deterministic model. Eq. 3 in Fuhrmann et al.\n"
  "            Pr  = u * Pv                         :Pr is calculated as Pv * u (running value of Use)\n"
  "            Pv   = Pv - u * Pv                   :update Pv as per Eq. 3 in Fuhrmann et al.\n"
  "            :printf(\"Pv = %g\\n\", Pv)\n"
  "            :printf(\"Pr = %g\\n\", Pr)\n"
  "                tsyn = t\n"
  "                if (ranfNMDA() < Pr){\n"
  "	   	        A = A + weight*factor\n"
  "	            B = B + weight*factor\n"
  "                }\n"
  "}    \n"
  "		\n"
  ;
#endif
