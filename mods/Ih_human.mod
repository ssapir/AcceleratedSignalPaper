:Comment :
:Reference : :		Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON	{
	SUFFIX Ih_human
	NONSPECIFIC_CURRENT ihcn
	RANGE gIhbar, gIh, ihcn 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gIhbar = 0.00001 (S/cm2) 
	ehcn =  -49.765 (mV)
}

ASSIGNED	{
	v	(mV)
	ihcn	(mA/cm2)
	gIh	(S/cm2)
	mInf
	mTau
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gIh = gIhbar*m
	ihcn = gIh*(v-ehcn)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
}

PROCEDURE rates(){
	UNITSOFF
        if(v == -90.963){
            v = v + 0.0001
        }
		mInf = 1/(1+exp((v+90.963)/8.0775))
		mTau = 0.00000000019419+1/(exp(-23.428-0.21756*v)+exp(-0.00000000013881+0.082329*v))
	UNITSON
}
