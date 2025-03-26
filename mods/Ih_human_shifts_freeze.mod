:Comment :
:Reference : :		Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON	{
	SUFFIX Ih_human_shifts_freeze
	NONSPECIFIC_CURRENT ihcn
	RANGE gIhbar, gIh, ihcn, m_alpha_shift_v, E_PAS
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gIhbar = 0.00001 (S/cm2) 
	ehcn =  -45.0 (mV)
	m_alpha_shift_v = -20 (mV)
	E_PAS = -70 (mV)
}

ASSIGNED	{
	ihcn	(mA/cm2)
	gIh	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gIh = gIhbar*m
	ihcn = gIh*(E_PAS-ehcn)
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
        if(-140 + m_alpha_shift_v > E_PAS){
            v = -140
        }
		mAlpha =  0.001*6.43*(E_PAS+154.9 + m_alpha_shift_v)/(exp((E_PAS+154.9 + m_alpha_shift_v)/11.9)-1)
		mBeta  =  0.001*193*exp(v/33.1)
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
