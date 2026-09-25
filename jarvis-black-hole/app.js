// ================= shaders =================
// Units: 1 = 490.08 km (Earth's radius is 13). The black-hole pair sits at the origin; everything
// else is placed around it at true scale, with real positions for the current moment.
const NL=4,NP=14,NB=9,NNET=10;
const KM=13/6371;
// the Moon's dark maria and ray craters (selenographic latitude, east longitude, diameter in km)
const MARIA=[[32.8,-15.6,1146],[28,17.5,707],[8.5,31.4,873],[3,24,380],[17,59.1,556],[-7.8,51.3,909],[-15.2,35.5,333],[-21.3,-16.6,715],[-24.4,-38.6,389],
  [-10,-23.1,376],[7.5,-30.9,513],[13.3,3.6,245],[2.4,1.7,335],[10.9,-8.8,290],[44.1,-31.5,236],[38,29.2,384],[13.3,86.1,420],[1.3,87.5,373],
  [6.8,68.4,243],[1.1,65.1,139],[22.6,67.7,150],[56.8,81.5,273],[-19.4,-92.8,294],[27.3,147.9,277],[-33.7,163.5,318],[-38.9,93,420],
  [18.4,-57.4,1300],[2,-48,900],[35,-45,800],[-10,-45,700],[25,-65,800],[58,-30,400],[56.5,-5,420],[56,20,420],[55,42,300],[51.6,-9.4,101],[-5.2,-68.6,173]];
const RAYC=[[-43.31,-11.36,1.],[9.62,-20.08,.7],[8.1,-38.,.45],[23.7,-47.4,.5]];
const sel=(la,lo)=>{la*=Math.PI/180;lo*=Math.PI/180;return[Math.cos(la)*Math.cos(lo),Math.cos(la)*Math.sin(lo),Math.sin(la)];};
const f3=a=>a.map(x=>x.toFixed(5)).join(",");
const MARIA_GLSL=`
const vec4 RAYS[4]=vec4[](${RAYC.map(([la,lo,k])=>`vec4(${f3(sel(la,lo))},${k.toFixed(2)})`).join(",")});`;

const VS=`#version 300 es
in vec2 p;out vec2 v;void main(){v=p*.5+.5;gl_Position=vec4(p,0.,1.);}`;

const SCENE=`#version 300 es
precision highp float;precision highp sampler3D;precision highp sampler2DArray;
in vec2 v;out vec4 o;
uniform vec2 R;uniform float T,DT,energy,voice,surge,frame,focal,trailGlow,mono,NETK,CORE,FIL;
uniform vec3 cam,hot,cool,ACC,SUNW;uniform mat3 view;uniform int STEPS,ASTEPS,ATM;
uniform sampler3D NZ;
uniform vec3 H1,H2;uniform float MH,BPH,BSEP,RINGA;
uniform vec4 LP[${NL*NP}],LB[${NL}],LC[${NL}];
uniform vec4 BC[${NB}],BX[${NB}],BM[${NB}],BS[${NB}],BP[${NB}];
uniform vec3 SUNC;uniform float SUNR;
uniform vec3 camE,ESUN,EMOON;uniform float EALT,CLD,MPH;uniform mat3 EMT,MMT;
uniform sampler2D EG,EN,EW,ECL,ETP,MOONM;uniform float GTEX,CTEX;
uniform sampler2DArray SAT;uniform vec4 SK[4],SE[4],SN[4];
uniform vec3 NETA[${NNET}],NETB[${NNET}];
const int NL=${NL},NP=${NP},NBOD=${NB},NNET=${NNET};
const float PI=3.14159265,TAU=6.2831853,RE=13.,KM=${KM.toFixed(10)};
const float HA=80.*KM,HC=6.*KM,HR=8.*KM,HM=1.2*KM,ESG=1.3,ESA=5.2;
const vec3 BRAY=vec3(2.8434,6.6445,16.2215),BOZ=vec3(.3185,.9218,.0417);   // per unit length at sea level (Bruneton 2017)
const float BMIE=1.9583,BMIEX=2.1737;
const float RB=39.,CBIN=6.4,CB1=15.5,MD0=1.1,MD1=1.72,LEMA=4.6,RINGR=36.,RINGH=1.1,JETL=260.;
${MARIA_GLSL}
const vec3 PCOL[${NB}]=vec3[](vec3(.8,.76,.7),vec3(1.,.95,.82),vec3(.72,.84,1.),vec3(.92,.92,.9),vec3(1.,.66,.46),vec3(1.,.93,.82),vec3(1.,.92,.74),vec3(.76,.93,.96),vec3(.6,.74,1.));
float PIX;

float nz(vec3 p){return texture(NZ,p).r;}
float hash13(vec3 p){p=fract(p*.1031);p+=dot(p,p.zyx+31.32);return fract((p.x+p.y)*p.z);}
float ign(vec2 p){return fract(52.9829189*fract(dot(p,vec2(.06711056,.00583715))));}
vec3 rainbow(float h){return clamp(abs(mod(h*6.+vec3(0.,4.,2.),6.)-3.)-1.,0.,1.);}
float lensF(float u){return u*(2.*u*u+3.)/pow(1.+u*u,1.5);}   // integral of the bending along a straight line (u = distance / impact parameter)
vec3 bend(vec3 p,vec3 rd,float a){vec3 q=rd*dot(p,rd)-p;float l=length(q);return l>1e-9?normalize(rd+q*(a/l)):rd;}
float sph(vec3 ro,vec3 rd,vec3 c,float r){vec3 oc=ro-c;float b=dot(oc,rd);vec3 pp=oc-rd*b;float h=r*r-dot(pp,pp);if(h<0.)return 1e30;h=sqrt(h);float t=-b-h;return t>0.?t:1e30;}
vec2 loc(vec3 d,vec3 C,float s){vec3 U=normalize(cross(vec3(0.,1.,0.),C)),V=cross(C,U);return vec2(dot(d,U),dot(d,V))/s;}
float winEdge(vec2 u){vec2 e=smoothstep(0.,.05,u)*smoothstep(1.,.95,u);return e.x*e.y;}
// light on a solid surface near the core: the Sun, the black holes' glowing gas, faint starlight
vec3 shade(vec3 p,vec3 n,vec3 alb,vec3 sun,float vis){
  float dd=length(p);vec3 dl=-p/max(dd,1e-3);
  vec3 L=vec3(1.,.96,.9)*1.05*max(dot(n,sun),0.)*vis+(mix(cool,hot,.65)+.25)*max(dot(n,dl),0.)*(16./(6.+dd*.5))+vec3(.010,.011,.015);
  return alb*L;
}
float segSeg(vec3 a,vec3 b,vec3 c,vec3 d,out float s,out float u){
  vec3 d1=b-a,d2=d-c,r=a-c;float A=dot(d1,d1),E=dot(d2,d2),F=dot(d2,r),C=dot(d1,r),B=dot(d1,d2),den=A*E-B*B;
  s=den>1e-7?clamp((B*F-C*E)/den,0.,1.):0.;
  u=(B*s+F)/max(E,1e-7);
  if(u<0.){u=0.;s=clamp(-C/max(A,1e-7),0.,1.);}
  else if(u>1.){u=1.;s=clamp((B-C)/max(A,1e-7),0.,1.);}
  vec3 q=a+d1*s-c-d2*u;return dot(q,q);
}
float ptSeg(vec3 x,vec3 a,vec3 b,out float s){vec3 ab=b-a;s=clamp(dot(x-a,ab)/max(dot(ab,ab),1e-7),0.,1.);vec3 d=a+ab*s-x;return dot(d,d);}
// finer octaves fade in only once they're bigger than a pixel, so detail grows as you fly closer
float fbmL(vec3 p,float fp,int oct){
  float f=0.,a=.5,fr=1.;
  for(int i=0;i<10;i++){if(i>=oct)break;float k=clamp((.03/fr-fp*1.5)/(fp*1.5),0.,1.);if(k<=0.)break;f+=a*(nz(p*fr+float(i)*.173)-.5)*k;fr*=2.03;a*=.5;}
  return f;
}
float craters(vec3 q,float fpq,out float rim){
  float h=0.,sc=5.;rim=0.;
  for(int i=0;i<6;i++){
    float size=.25/sc;if(size<fpq*2.5)break;float vis=clamp(size/(fpq*2.5)-1.,0.,1.);
    vec3 c=q*sc,id=floor(c)+float(i)*17.,fr=fract(c);
    if(hash13(id)>.45){
      vec3 ctr=vec3(hash13(id+1.3),hash13(id+2.9),hash13(id+4.1))*.5+.25;float rr=length(fr-ctr)/(.1+.22*hash13(id+5.7));
      if(rr<1.5){h+=(rr<1.?(rr*rr-1.)*.6:0.)*vis/(1.+float(i)*.3);rim+=exp(-(rr-1.)*(rr-1.)/.015)*vis*.5;}
    }
    sc*=2.9;
  }
  return h;
}

// ================= the black holes =================
vec3 thermal(float To,float g){
  vec3 base=mix(cool,hot,smoothstep(.25,.95,To));
  base=mix(base,vec3(1.,.97,.93),smoothstep(.85,1.35,To)*.8);
  return base*mix(vec3(1.12,.9,.8),vec3(.88,.96,1.14),smoothstep(.7,1.35,g));
}
// orbiting, turbulent gas: sampled on a circle (no seam), two flow phases cross-faded; DT is the disk's own clock
float gas(float rr,float ang,float y,float om,float seed){
  float P=7.,ph=fract(DT/P),ph2=fract(DT/P+.5),w=abs(ph*2.-1.);
  float a1=ang-om*ph*P,a2=ang-om*ph2*P;
  float s1=sin(a1),k1=cos(a1),s2=sin(a2),k2=cos(a2);
  float big=smoothstep(.3,.72,mix(nz(vec3(k1*.2,s1*.2,rr*.23+y*.24+seed)),nz(vec3(k2*.2,s2*.2,rr*.23+y*.24+seed)+.37),w));
  if(big<.01)return 0.;
  float fine=mix(nz(vec3(k1*.45,s1*.45,rr*1.44+seed)),nz(vec3(k2*.45,s2*.45,rr*1.44+seed)+.61),w);
  return big*(.6+.8*smoothstep(.3,.75,fine));
}
// distance to the figure-eight stream (lemniscate of Bernoulli, locked to the binary), in the binary's frame
float lem(vec3 p,out vec2 q,out vec2 gF){
  float c=cos(BPH),s=sin(BPH);q=vec2(c*p.x+s*p.z,-s*p.x+c*p.z);
  float A2=LEMA*LEMA,r2=dot(q,q),F=r2*r2-A2*(q.x*q.x-q.y*q.y);
  gF=vec2(4.*q.x*r2-2.*A2*q.x,4.*q.y*r2+2.*A2*q.y);
  return abs(F)/max(length(gF),A2*.8);
}
// the AI's collector ring: a band of panels facing the gas
vec3 ringShade(vec3 hp,vec3 rd,out float alpha){
  float ang=atan(hp.z,hp.x)-RINGA,yv=hp.y/RINGH,pn=ang/TAU*96.,u=fract(pn);
  vec3 nr=normalize(vec3(hp.x,0.,hp.z));bool inner=dot(rd,nr)>0.;vec3 n=inner?-nr:nr;
  float fr=max(smoothstep(.04,0.,min(u,1.-u)),smoothstep(.9,1.,abs(yv)));
  float cell=hash13(vec3(floor(pn),floor(yv*3.),7.));
  vec3 alb=inner?vec3(.03,.04,.065)*(.8+.4*cell):vec3(.075,.08,.085)*(.8+.3*cell);
  vec3 c=shade(hp,n,alb,SUNW,1.);
  vec3 dl=-normalize(hp),hh=normalize(dl-rd);
  c+=(mix(cool,hot,.7)+.2)*pow(max(dot(n,hh),0.),60.)*(inner?.55:.08);
  float flow=pow(.5+.5*sin(ang*6.-T*(2.+3.*energy)),8.);
  c+=ACC*fr*(.22+.9*flow+energy*.6+voice*.8);
  float nd=abs(fract(ang/TAU*8.+.5)-.5)*TAU/8.*RINGR;
  c+=ACC*exp(-(nd*nd+hp.y*hp.y*.3)*5.)*(2.+2.*flow);
  alpha=fr>.5?1.:.93;
  return c;
}
// the jet collectors: four rings the jets are threaded through (world coordinates)
float hoopHit(vec3 ro,vec3 rd,float tmax,out vec3 col){
  col=vec3(0.);float best=tmax;
  if(abs(rd.y)<1e-6)return 1e30;
  for(int i=0;i<4;i++){
    float Y=i<2?42.:88.,r0=i<2?2.8:5.2,r1=i<2?3.7:6.4;if(i==1||i==3)Y=-Y;
    float t=(Y-ro.y)/rd.y;if(t<=0.||t>=best)continue;
    vec3 hp=ro+rd*t;float rr=length(hp.xz);if(rr<r0||rr>r1)continue;
    best=t;float u=(rr-r0)/(r1-r0),ang=atan(hp.z,hp.x);
    vec3 n=vec3(0.,-sign(rd.y),0.);
    vec3 c=shade(hp,n,vec3(.07,.075,.08)*(.85+.3*hash13(vec3(floor(ang*9.),float(i),3.))),SUNW,1.);
    float seam=smoothstep(.06,0.,abs(fract(ang/TAU*24.)-.5)-.44);
    c+=ACC*(exp(-u*u*50.)*2.6+exp(-(1.-u)*(1.-u)*120.)*.5+seam*.35)*(.75+.25*sin(ang*3.+T*4.))*(.8+energy*.8+voice*1.2);
    col=c;
  }
  return best<tmax?best:1e30;
}
// relativistic jets, braided by the orbiting pair near the base (closed-form line integral, never aliased)
vec3 jets(vec3 ro,vec3 rd,float tmax){
  vec2 a=ro.xz,b=rd.xz;float bb=dot(b,b);if(bb<1e-5)return vec3(0.);
  float ts=-dot(a,b)/bb;if(ts<0.||ts>tmax)return vec3(0.);   // the jet's nearest point is behind you, or hidden behind what the ray hit
  vec3 pc=ro+rd*ts;float ay=abs(pc.y);
  if(ay>JETL||ay<.3)return vec3(0.);
  float w=.18+.028*ay,fp=ts*PIX,we=sqrt(w*w+fp*fp*.8);
  float rh=BSEP*.5*exp(-ay/14.),ph=BPH-ay*.12,I=0.;
  for(int k=0;k<2;k++){
    float a0=ph+float(k)*PI;vec2 sc=rh*vec2(cos(a0),sin(a0));
    vec2 dd=a+b*(-dot(a-sc,b)/bb)-sc;I+=exp(-dot(dd,dd)/(we*we))*w/we;
  }
  float cz=-sign(pc.y)*rd.y,dop=pow(1./(1.0483*(1.-.3*cz)),2.);          // relativistic beaming: the approaching jet is brighter
  float knots=.55+.45*pow(.5+.5*sin(ay*.32-T*7.),5.);
  float fall=exp(-ay/(JETL*.33))*smoothstep(.3,2.2,ay);
  vec3 jc=mix(vec3(.55,.72,1.),vec3(.85,.92,1.),exp(-ay/20.));
  jc=mix(jc,ACC*.9+.08,.45*smoothstep(38.,46.,ay));                        // past the first collector it carries the AI's colour
  return jc*I*knots*fall*dop*(1.1+energy*.8+voice*1.5+surge)*.8/max(sqrt(bb),.25)*mix(1.,.4,smoothstep(120.,2000.,ts));
}
// the AI's filaments: power and data running out from the ring to the Earth, the Moon and deep space
vec3 filaments(vec3 ro,vec3 rd,float tmax){
  vec3 c=vec3(0.);float tm=min(tmax,2e5);
  for(int i=0;i<NNET;i++){
    vec3 A=NETA[i],B=NETB[i];float s,u;
    float d2=segSeg(ro,ro+rd*tm,A,B,s,u),t=s*tm,fp=t*PIX,w=.03,we2=w*w+fp*fp*.7;
    float g=exp(-d2/we2)*max(w*inversesqrt(we2),.07)*smoothstep(6.,60.,t);
    if(g<1e-4)continue;
    float L=length(B-A),al=u*L;
    float pulse=pow(fract(al/38.-T*.9+float(i)*.37),14.)*3.;
    c+=ACC*g*(.3+pulse)*(i<8?exp(-al/2600.):1.);
    if(i==8){float e2=ptSeg(B,ro,ro+rd*tm,s);float te=s*tm,fe=te*PIX;c+=ACC*exp(-e2/(.004+fe*fe*2.))*1.1*smoothstep(6.,60.,te);}   // relay satellites at the ends
  }
  return c*(.6+energy*.6+voice*.9);
}

// ================= rocks near the core =================
float asteroids(vec3 ro,vec3 rd,float tmax,out vec3 col){
  col=vec3(0.);float tc=length(ro-cam);
  float an=T*.012,ca=cos(an),sa=sin(an);mat2 M=mat2(ca,sa,-sa,ca);
  ro.xz=M*ro.xz;rd.xz=M*rd.xz;vec3 sun=SUNW;sun.xz=M*sun.xz;
  const float BH=4.,RO=74.,CS=3.;
  float t0=0.,t1=tmax;
  if(abs(rd.y)>1e-5){float ta=(-BH-ro.y)/rd.y,tb=(BH-ro.y)/rd.y;t0=max(t0,min(ta,tb));t1=min(t1,max(ta,tb));}
  else if(abs(ro.y)>BH)return 1e30;
  float a=dot(rd.xz,rd.xz);
  if(a>1e-8){float b=dot(ro.xz,rd.xz),c=dot(ro.xz,ro.xz)-RO*RO,h=b*b-a*c;if(h<0.)return 1e30;h=sqrt(h);t0=max(t0,(-b-h)/a);t1=min(t1,(-b+h)/a);}
  else if(dot(ro.xz,ro.xz)>RO*RO)return 1e30;
  if(t0>=t1)return 1e30;
  vec3 p=ro+rd*t0,cell=floor(p/CS);
  vec3 rdd=vec3(abs(rd.x)<1e-6?1e-6:rd.x,abs(rd.y)<1e-6?1e-6:rd.y,abs(rd.z)<1e-6?1e-6:rd.z);
  vec3 st=sign(rdd),tDel=abs(CS/rdd),tMax=t0+((cell+step(0.,st))*CS-p)/rdd;
  for(int i=0;i<96;i++){
    if(i>=ASTEPS)break;
    vec3 cc=(cell+.5)*CS;float rc=length(cc.xz);
    float dens=exp(-(rc-54.)*(rc-54.)/150.)*exp(-cc.y*cc.y/5.);
    if(hash13(cell+.71)<dens*.2){
      float rad=mix(.1,.75,pow(hash13(cell+3.1),3.));
      vec3 c=cc+(vec3(hash13(cell+5.2),hash13(cell+7.9),hash13(cell+2.4))-.5)*(CS-2.2*rad);
      vec3 ax=rad*vec3(1.,.55+.4*hash13(cell+9.3),.7+.3*hash13(cell+1.7));
      vec3 oc=(ro-c)/ax,dd=rd/ax;float A=dot(dd,dd),B=dot(oc,dd),C=dot(oc,oc)-1.,D=B*B-A*C;
      if(D>0.){
        float th=(-B-sqrt(D))/A;
        if(th>0.&&th<tmax){
          vec3 hp=ro+rd*th,q=(hp-c)/ax,n=normalize(q/ax),np=q*.35+cell*.173;float b0=nz(np);
          vec3 g=vec3(nz(np+vec3(.03,0.,0.)),nz(np+vec3(0.,.03,0.)),nz(np+vec3(0.,0.,.03)))-b0;
          float fpq=(th+tc)*PIX/rad,rim,cr=craters(normalize(q)+cell*.013,fpq,rim),dt2=fbmL(q*1.5+cell*.29,fpq*1.5,7);
          n=normalize(n-g*4.-q*cr*.35);
          vec3 tint=mix(vec3(.62,.54,.46),vec3(.52,.51,.54),hash13(cell+6.6))*(.14+.16*hash13(cell+4.4));
          col=shade(hp,n,tint*(.75+.5*b0)*(1.+dt2*.8)*(1.+cr*.4)+rim*.03,sun,1.)*vec3(1.,.97,.95);
          return th;
        }
      }
    }
    if(tMax.x<tMax.y&&tMax.x<tMax.z){if(tMax.x>t1)break;cell.x+=st.x;tMax.x+=tDel.x;}
    else if(tMax.y<tMax.z){if(tMax.y>t1)break;cell.y+=st.y;tMax.y+=tDel.y;}
    else{if(tMax.z>t1)break;cell.z+=st.z;tMax.z+=tDel.z;}
  }
  return 1e30;
}

// ================= the solar system (camera-relative, true scale) =================
// oblate planet: equatorial radius BC.w, flattening BX.w along the pole BX.xyz
float ellHit(vec3 ro,vec3 rd,int i,out vec3 n){
  vec3 ax=BX[i].xyz;float k=1./(1.-BX[i].w)-1.;
  vec3 oc=ro-BC[i].xyz,o2=oc+ax*(dot(oc,ax)*k),d2=rd+ax*(dot(rd,ax)*k);
  float a=dot(d2,d2),b=dot(o2,d2);vec3 pp=o2-d2*(b/a);float r=BC[i].w,h=r*r-dot(pp,pp);
  if(h<0.)return 1e30;
  float t=(-b-sqrt(h*a))/a;if(t<=0.)return 1e30;
  vec3 q=o2+d2*t;n=normalize(q+ax*(dot(q,ax)*k));return t;
}
float ringTau(float x){   // Saturn's rings: optical depth against radius (in Saturn radii), D, C, B, Cassini Division, A with the Encke gap, F
  float tau=0.;
  if(x<1.11||x>2.34)return 0.;
  if(x<1.236)tau=.012;
  else if(x<1.527)tau=.08+.04*sin(x*420.);
  else if(x<1.951)tau=1.2+.9*smoothstep(1.53,1.62,x)+.45*sin(x*260.+1.);
  else if(x<2.027)tau=.08+.05*smoothstep(1.99,2.02,x);
  else if(x<2.269)tau=.45+.15*sin(x*330.);
  tau*=1.-.95*exp(-pow((x-2.2135)/.0028,2.));
  return tau+.6*exp(-pow((x-2.326)/.0016,2.));
}
vec4 satRing(vec3 ro,vec3 rd,out float t){
  t=1e30;vec3 ax=BX[6].xyz;float dn=dot(rd,ax);if(abs(dn)<1e-8||BC[6].w<=0.)return vec4(0.);
  float tt=dot(BC[6].xyz-ro,ax)/dn;if(tt<=0.)return vec4(0.);
  vec3 hp=ro+rd*tt;float x=length(hp-BC[6].xyz)/BC[6].w,tau=ringTau(x);
  if(tau<=0.)return vec4(0.);
  float mu=abs(dn),a=1.-exp(-tau/max(mu,.01));if(a<.002)return vec4(0.);
  vec3 sd=BS[6].xyz;float mu0=abs(dot(sd,ax));bool lit=dot(sd,ax)*dot(-rd,ax)>0.;
  vec3 alb=x<1.527?vec3(.42,.4,.37):vec3(.78,.71,.6);
  float I=lit?(1.-exp(-tau*(1./max(mu0,.01)+1./max(mu,.01))))*mu0/(mu0+mu)*1.35:tau/max(mu,.05)*exp(-tau/max(mu0,.02))*exp(-tau/max(mu,.02))*1.2;
  float sh=sph(hp,sd,BC[6].xyz,BC[6].w)<1e29?.03:1.;
  t=tt;return vec4(alb*BS[6].w*ESG*I*sh,a);
}
float blob(vec3 bf,float la,float lo,float r){la=radians(la);lo=radians(lo);vec3 c=vec3(cos(la)*cos(lo),cos(la)*sin(lo),sin(la));return smoothstep(cos(radians(r)),cos(radians(r*.45)),dot(bf,c));}
vec3 bodyShade(int i,vec3 n,vec3 rd,float t){
  vec3 sd=BS[i].xyz,pole=BX[i].xyz,pm=BM[i].xyz,ey=cross(pole,pm);
  vec3 bf=vec3(dot(n,pm),dot(n,ey),dot(n,pole));
  float lat=asin(clamp(bf.z,-1.,1.)),la=degrees(lat),mu=max(dot(n,-rd),0.),fp=t*PIX/max(mu,.12)/BC[i].w,ndl=dot(n,sd);
  vec3 alb;float E=BS[i].w*ESG,gl=1.;
  if(i==0){float rim,cr=craters(bf,fp,rim),d=fbmL(bf*3.,fp*3.,8);alb=vec3(.16,.145,.13)*(1.+d*.8)*(1.+cr*.45)+rim*.05;}
  else if(i==1){float f=fbmL(vec3(bf.xy*1.3,la*.04)+vec3(T*.001,0.,0.),fp,5);alb=vec3(.93,.87,.72)*(.9+f*.35);gl=.85;}
  else if(i==4){   // Mars: dark albedo features at their real places, bright Hellas and polar caps
    float d=fbmL(bf*2.5,fp*2.5,8);
    float dk=blob(bf,10.,69.,11.)*.5+blob(bf,47.,-23.,12.)*.4+blob(bf,-25.,-40.,15.)*.35+blob(bf,-5.,8.,13.)*.35+blob(bf,-22.,145.,15.)*.35
            +blob(bf,-32.,-155.,13.)*.3+blob(bf,-15.,100.,11.)*.3+blob(bf,48.,118.,13.)*.2+blob(bf,-13.,-60.,5.)*.3;
    float br=blob(bf,-42.4,70.5,12.)*.35+blob(bf,20.,5.,18.)*.12+blob(bf,18.6,-133.8,6.)*.15;
    alb=vec3(.74,.4,.21)*(1.-dk*(.75+d))*(1.+br+d*.5);
    alb=mix(alb,vec3(.92,.92,.95),smoothstep(79.,83.,abs(la)+d*6.));
  }else if(i==5){  // Jupiter: belts and zones at their real latitudes, the Great Red Spot at 22°S
    float lon=atan(bf.y,bf.x),turb=fbmL(vec3(lon*2.,la*.12,0.)+vec3(T*.004,0.,0.),fp*2.,8),x=la+turb*5.;
    float belt=smoothstep(6.,8.5,x)*smoothstep(19.,16.5,x)+smoothstep(-7.,-9.5,x)*smoothstep(-21.,-18.5,x)+.6*smoothstep(23.,26.,x)*smoothstep(32.,29.,x)
              +.5*smoothstep(-26.,-29.,x)*smoothstep(-35.,-32.,x)+.35*smoothstep(38.,45.,abs(x));
    alb=mix(vec3(.93,.87,.76),vec3(.6,.42,.3),clamp(belt,0.,1.));
    vec2 gd=vec2((lon-.8)*cos(lat)/.13,(la+22.5)/5.);float gr=length(gd);
    alb=mix(alb,vec3(.8,.42,.3)*(.9+.2*fbmL(vec3(gd,T*.05),fp*6.,5)),smoothstep(1.2,.6,gr));
    gl=.7;
  }else if(i==6){  // Saturn: soft bands, the north polar hexagon
    float lon=atan(bf.y,bf.x),turb=fbmL(vec3(lon*2.,la*.1,1.),fp*2.,6),x=la+turb*3.;
    alb=mix(vec3(.9,.83,.63),vec3(.77,.65,.45),.5+.5*sin(x*.35));
    alb=mix(alb,vec3(.58,.62,.62),smoothstep(72.,76.,la)*.6);
    float dn=dot(sd,pole);if(abs(dn)>1e-3){vec3 hp=BC[i].xyz+n*BC[i].w;float tt=dot(BC[i].xyz-hp,pole)/dn;if(tt>0.)gl*=exp(-ringTau(length(hp+sd*tt-BC[i].xyz)/BC[i].w)/max(abs(dn),.02));}
  }else if(i==7){alb=vec3(.62,.82,.86)*(.95+.05*cos(lat*4.));}
  else{float f=fbmL(vec3(bf.xy*2.,la*.08),fp*2.,5);alb=vec3(.27,.43,.88)*(.9+.2*sin(la*.3+f*4.));alb=mix(alb,vec3(.1,.18,.45),blob(bf,-20.,40.,6.));}
  vec3 c=alb*E*max(ndl,0.)*gl;
  if(i>=5)c*=mix(.55,1.,sqrt(mu));   // limb darkening on the giants
  return c;
}
vec3 moonS(vec3 n,vec3 rd,float t){
  vec3 q=MMT*n;float mu=max(dot(n,-rd),0.),fp=t*PIX/max(mu,.12)/BC[3].w,mare=0.;
  float mla=asin(clamp(q.z,-1.,1.)),mlo=atan(q.y,q.x);
  mare=textureLod(MOONM,vec2(mlo/TAU+.5,.5-mla/PI),1.5).r*(.8+.4*nz(q*3.1+.3))+(nz(q*9.7)-.5)*.1;mare=clamp(mare,0.,1.);
  float rim,cr=craters(q,fp,rim),d=fbmL(q*3.,fp*3.,8);
  float ray=0.;
  for(int i=0;i<4;i++){vec3 c=RAYS[i].xyz;float th=acos(clamp(dot(q,c),-1.,1.));if(th<.9){vec3 tg=normalize(q-c*dot(q,c));vec3 e1=normalize(cross(c,vec3(0.,0.,1.))),e2=cross(c,e1);float az=atan(dot(tg,e2),dot(tg,e1));
    ray+=RAYS[i].w*(pow(nz(vec3(az*2.6,RAYS[i].w*3.,.5)),5.)*2.5*exp(-th*3.)+exp(-th*th*900.)*1.5);}}
  vec3 alb=vec3(mix(.125,.075,mare))*(1.+d*.6)*(1.+cr*.4)*vec3(1.,.985,.96)+rim*.04+ray*.06;
  vec3 sd=BS[3].xyz;float mu0=max(dot(n,sd),0.);
  float ph=acos(clamp(dot(sd,-rd),-1.,1.));
  vec3 c=alb*BS[3].w*ESG*(2.*mu0/(mu0+mu+1e-3)*.8+mu0*.2)*(1.+.35*exp(-ph/.08));   // Lommel–Seeliger with the opposition surge
  vec3 te=normalize(BC[2].xyz-BC[3].xyz);c+=alb*vec3(.55,.7,1.)*.02*max(dot(n,te),0.)*(.5-.5*dot(sd,te));   // earthshine
  return c;
}

// ================= Earth =================
// sunlight's optical depth / scale height from radius r toward a sun at cos-zenith cz (Schüler's Chapman approximation)
float odep(float r,float cz,float H){
  float X=r/H,c=sqrt(1.5707963*X),h=(r-RE)/H;
  if(cz>=0.)return exp(-h)*c/((c-1.)*cz+1.);
  float x0=X*sqrt(max(0.,1.-cz*cz)),c0=sqrt(1.5707963*x0);
  return 2.*c0*exp(RE/H-x0)-exp(-h)*c/((c-1.)*(-cz)+1.);
}
vec3 sunTrans(float r,float cz){
  float sh=sqrt(max(0.,1.-RE*RE/(r*r)));if(cz<-sh)return vec3(0.);      // the Earth is in the way
  float oR=odep(r,cz,HR),oM=odep(r,cz,HM);
  float am=oR/max(exp(-(r-RE)/HR),1e-9);                                  // air mass, for the ozone layer
  return exp(-(BRAY*HR*oR+BMIEX*HM*oM+BOZ*15.*KM*min(am,40.)*step(r,RE+40.*KM)));
}
// single scattering along the view ray (origin o relative to Earth's centre), Rayleigh + Mie + ozone
vec3 atmos(vec3 o,vec3 rd,float t0,float t1,vec3 sd,bool ground,float jit,out vec3 Tv){
  vec3 tau=vec3(0.),LR=vec3(0.),LM=vec3(0.);float mu=dot(rd,sd),L=t1-t0;
  for(int i=0;i<24;i++){
    if(i>=ATM)break;
    float u0=float(i)/float(ATM),u1=float(i+1)/float(ATM),um=(float(i)+jit)/float(ATM);
    // bunch the samples toward the ground end, where the air is densest
    float ta=ground?t0+L*(1.-(1.-u0)*(1.-u0)):t0+L*u0,tb=ground?t0+L*(1.-(1.-u1)*(1.-u1)):t0+L*u1,t=ground?t0+L*(1.-(1.-um)*(1.-um)):t0+L*um,ds=tb-ta;
    vec3 q=o+rd*t;float r=length(q),h=r-RE;
    float dR=exp(-h/HR),dM=exp(-h/HM),dO=max(0.,1.-abs(h-25.*KM)/(15.*KM));
    vec3 ext=BRAY*dR+BMIEX*dM+BOZ*dO;
    vec3 Tt=exp(-(tau+ext*ds*.5))*sunTrans(r,dot(q,sd)/r);
    LR+=Tt*dR*ds;LM+=Tt*dM*ds;tau+=ext*ds;
  }
  Tv=exp(-tau);
  const float g=.76;float pR=.0596831*(1.+mu*mu),pM=.1193662*(1.-g*g)*(1.+mu*mu)/((2.+g*g)*pow(1.+g*g-2.*g*mu,1.5));
  return ESA*(LR*BRAY*pR+LM*BMIE*pM);
}
float cloud(vec3 q,float fp){
  float lat=asin(clamp(q.y,-1.,1.)),lon=atan(-q.z,q.x);
  float v=textureLod(ECL,vec2(lon/TAU+.5,.5-lat/PI),log2(max(fp/CTEX,1e-4))).r;
  float c=smoothstep(.05,.85,v);
  float k=smoothstep(CTEX*1.2,CTEX*.12,fp);           // closer than the map resolves: grow cumulus-scale structure
  if(k>0.&&c>0.){vec3 w=q*420.;float n=nz(w)*.55+nz(w*2.13+.3)*.3+nz(w*4.7+.6)*.15;c=clamp(c*(1.+k*(n-.45)*1.8)-k*.12*(1.-c),0.,1.);}
  return c;
}
vec3 earthSurface(vec3 ro,vec3 rd,float t,bool prim){
  vec3 rdE=EMT*rd,pE=prim?EMT*camE+rdE*t:EMT*(ro-BC[2].xyz+rd*t),q=normalize(pE);
  float lat=asin(clamp(q.y,-1.,1.)),lon=atan(-q.z,q.x);vec2 ll=vec2(lon/TAU+.5,.5-lat/PI);
  float mu=max(dot(q,-rdE),0.),fp=t*PIX/max(mu,.15),lod=log2(max(fp/GTEX,1e-4));
  // imagery: NASA Blue Marble everywhere, replaced by Esri World Imagery (to ~0.3 m) wherever it has streamed in
  vec3 day=textureLod(EG,ll,lod).rgb;float sat=0.;
  if(prim)for(int k=3;k>=0;k--){
    if(SN[k].w<=0.)continue;
    vec3 dq=SK[k].xyz+rdE*t;vec2 uv=vec2(dot(dq,SE[k].xyz),dot(dq,SN[k].xyz))/(2.*SE[k].w)+.5;
    if(uv.x<=0.||uv.y<=0.||uv.x>=1.||uv.y>=1.)continue;
    vec4 s=textureLod(SAT,vec3(uv,SK[k].w),log2(max(fp/SN[k].w,1e-4)));
    float a=s.a*winEdge(uv);if(a<.004)continue;
    day=mix(day,s.rgb/max(s.a,1e-3),a);sat=max(sat,a);
  }
  vec3 alb=pow(day,vec3(2.2));
  {float lw=dot(alb,vec3(.2126,.7152,.0722));alb=max(mix(vec3(lw),alb,1.+.14*(1.-sat)),0.);}   // Blue Marble reads a little washed out: restore some colour
  float water=textureLod(EW,ll,lod).r*(1.-sat*.9);
  vec3 E=normalize(vec3(q.z,0.,-q.x)+vec3(1e-6,0.,0.)),N=cross(q,E),sE=ESUN;
  // terrain relief from the elevation map, for the view from orbit (the imagery already carries it up close)
  vec3 nq=q;float rel=(1.-sat)*(1.-water)*smoothstep(.0003,.003,fp);
  if(rel>0.){vec2 tx=vec2(1./2048.,1./1024.);
    float hx=textureLod(ETP,ll+vec2(tx.x,0.),0.).r-textureLod(ETP,ll-vec2(tx.x,0.),0.).r,hy=textureLod(ETP,ll-vec2(0.,tx.y),0.).r-textureLod(ETP,ll+vec2(0.,tx.y),0.).r;
    nq=normalize(q-(E*hx/max(cos(lat),.15)+N*hy)*1.6*rel);}
  float cz=dot(q,sE),ndl=dot(nq,sE);
  vec3 Ts=sunTrans(RE,cz),skyL=vec3(.04,.07,.12)*smoothstep(-.2,.35,cz);
  vec3 col=alb*(ESG*Ts*max(ndl,0.)+skyL);
  if(water>.01){   // ocean: sun glint on the waves, and the sky's reflection toward the horizon
    vec3 hh=normalize(sE-rdE);float rough=mix(.06,.3,smoothstep(1e-5,4e-3,fp)),a2=rough*rough,nh=max(dot(q,hh),0.),dd=nh*nh*(a2-1.)+1.;
    float D=a2/(PI*dd*dd),F=.02+.98*pow(1.-max(dot(hh,-rdE),0.),5.),nv=max(mu,.05),nl=max(dot(q,sE),0.);
    col+=water*Ts*ESG*min(D*F*nl/(4.*nv*max(nl,.05)),60.)*.9;
    col+=water*(.02+.98*pow(1.-nv,5.))*vec3(.05,.11,.22)*smoothstep(-.2,.3,cz);
    col+=water*(1.-sat)*vec3(.002,.011,.018)*ESG*Ts*max(ndl,0.);   // light scattered back up from under the surface
  }
  float night=smoothstep(.06,-.1,cz);
  vec3 lights=vec3(0.);
  if(night>0.){float Lt=textureLod(EN,ll,lod-.35).r,Lb=textureLod(EN,ll,lod+2.).r;lights=mix(vec3(1.,.66,.36),vec3(1.,.88,.74),smoothstep(.45,1.,Lt))*(pow(Lt,1.5)*2.2+Lb*.25)*night;}   // sodium-orange suburbs, whiter city cores
  // moonlight on the night side, from the Moon's real position and phase (brighter than the eye sees, like a long exposure)
  float ml=MPH*night*smoothstep(-.03,.08,dot(q,EMOON));
  if(ml>0.){col+=alb*vec3(.6,.68,.88)*.055*ml*max(dot(nq,EMOON),0.);
    if(water>.01){vec3 hm=normalize(EMOON-rdE);float nh=max(dot(q,hm),0.),dm=nh*nh*(-.985)+1.;col+=water*ml*vec3(.72,.8,1.)*min(.015/(PI*dm*dm)*.05,1.2)*max(dot(q,EMOON),0.);}}
  // clouds float ~6 km up: look them up where the view ray crosses that height, so they shift against the ground
  float cd=0.;vec3 ccol=vec3(0.);
  float cf=prim?smoothstep(.016,.045,EALT):1.;
  if(CLD>0.&&cf>0.){
    vec3 o=prim?camE:ro-BC[2].xyz;float bc=dot(o,rd),cc=prim?(EALT-HC)*(EALT+HC+2.*RE):dot(o,o)-(RE+HC)*(RE+HC),hc=bc*bc-cc;
    float tc=(hc>0.&&cc>0.)?cc/(-bc+sqrt(hc)):t;
    vec3 qc=normalize(prim?EMT*camE+rdE*tc:EMT*(o+rd*tc));
    cd=cloud(qc,fp)*cf*CLD;
    float cs=cloud(normalize(q+sE*(HC/max(cz,.08))/RE),fp*2.)*cf*CLD;   // cloud shadows, long at sunrise and sunset
    col*=1.-.6*cs;lights*=1.-.55*cs;
    if(cd>.002){
      float czc=dot(qc,sE);vec3 Tc=sunTrans(RE+HC,czc);
      vec3 st=normalize(sE-qc*czc+vec3(1e-6));float ahead=cloud(normalize(qc+st*.0022),fp);
      float relief=clamp(1.-(ahead-cd)*1.7,.42,1.25);                      // tops facing the sun are brighter, the far sides shaded
      float lit=max((czc+.14)/1.14,0.);                                    // thick cloud scatters light past the terminator
      ccol=vec3(1.,.995,.99)*ESG*.9*Tc*pow(lit,.8)*relief*(.7+.3*cd)+vec3(.045,.07,.12)*smoothstep(-.2,.3,czc)*cd+vec3(.004,.005,.008)+vec3(.036,.041,.052)*MPH*max(dot(qc,EMOON),0.)*smoothstep(.08,-.12,czc);
    }
  }
  col+=lights*(1.-cd*.85)+lights*cd*.18;                                   // city light glows through thin cloud
  col=mix(col,ccol,cd);
  // auroras round the geomagnetic poles on the night side
  vec3 mgN=vec3(.0484,.9867,.1551);float gm=abs(dot(q,mgN));   // geomagnetic poles, 80.65°N 72.68°W and its antipode
  col+=mix(vec3(.12,1.,.42),vec3(.8,.25,.5),smoothstep(.92,.96,gm))*exp(-(gm-.915)*(gm-.915)/.0005)*(.2+.9*pow(nz(vec3(q.x*3.,q.z*3.,T*.08)+q.y*.5),2.))*night*.9;
  return col;
}
vec3 earthAtmos(vec3 ro,vec3 rd,bool prim,float tb,bool ground,vec3 c){
  vec3 o=prim?camE:ro-BC[2].xyz;float b=dot(o,rd),cA=prim?(EALT-HA)*(EALT+HA+2.*RE):dot(o,o)-(RE+HA)*(RE+HA),h=b*b-cA;
  if(h<=0.)return c;
  h=sqrt(h);float t0=cA>0.?cA/(-b+h):0.,t1=min(-b+h,tb);
  if(cA>0.&&b>0.)return c;
  if(t1<=t0)return c;
  vec3 Tv;vec3 L=atmos(o,rd,t0,t1,ESUN*EMT,ground&&t1>=tb-1e-6*tb,ign(gl_FragCoord.xy+frame*3.7),Tv);
  return c*Tv+L;
}

// ================= the sky =================
vec3 starCol(float k){return k<.07?vec3(.6,.72,1.35):k<.3?vec3(.8,.88,1.1):k<.68?vec3(1.,.97,.9):k<.9?vec3(1.12,.82,.58):vec3(1.15,.6,.4);}
const vec3 HXd=vec3(.339,-.139,-.931),PLd=vec3(.814,-.342,.470),CRd=vec3(-.815,.1045,-.570),GXd=vec3(-.0797,-.4067,-.9100),CMd=vec3(-.945,-.208,.253);
vec3 helix(vec2 u){
  float r=length(u);if(r>1.35)return vec3(0.);
  float n1=nz(vec3(u*.8,.13)),n2=nz(vec3(u*2.1,.57)),rw=r+(n1-.5)*.2;
  float outer=exp(-(rw-.8)*(rw-.8)/.03)*(.5+.9*n2),inner=exp(-(rw-.55)*(rw-.55)/.016)*(.4+n1),core=exp(-r*r*5.)*.3;
  float kn=pow(nz(vec3(u/max(r,1e-3)*2.2,r*2.5+.3)),6.)*smoothstep(.4,.5,r)*smoothstep(.75,.6,r)*6.;
  return vec3(1.,.28,.14)*outer+vec3(.2,.9,.82)*inner+vec3(.3,.45,1.)*core+vec3(1.,.6,.38)*kn+vec3(.75,.85,1.)*exp(-r*r*1400.)*4.;
}
vec4 pillars(vec2 u){
  float r2=dot(u,u);if(r2>1.6)return vec4(0.);
  float edge=smoothstep(1.6,.9,r2),n=nz(vec3(u*.6,.21)),n2=nz(vec3(u*1.7,.77)),n3=nz(vec3(u*4.,.39));
  float cl=(.25+smoothstep(.2,.75,n))*exp(-r2*1.1)*edge;
  vec3 gs=mix(vec3(.08,.6,.55),vec3(1.,.62,.26),smoothstep(-.6,.8,u.y+(n2-.5)*.9));
  float m=0.,rim=0.;
  for(int i=0;i<3;i++){
    float fi=float(i),x0=-.38+fi*.36,top=.05+.32*fract(fi*.618+.2),w=.11+.035*fi;
    float y=clamp(u.y,-1.4,top),lean=(y-top)*.12*(fi-1.),wig=(nz(vec3(y*.55,fi*.31,.45))-.5)*.16;
    float taper=w*(1.25-.55*(y+1.4)/(top+1.4));
    float dd=length(vec2(u.x-x0-wig-lean,(u.y-y)*1.3))-taper+(n2-.5)*.05+(n3-.5)*.025;
    m=max(m,smoothstep(.02,-.03,dd));rim+=exp(-abs(dd+.004)*45.)*(.3+smoothstep(-.35,.1,u.y-top+.3));
  }
  m*=edge;rim*=edge;
  return vec4(gs*cl*(1.-m*.93)+vec3(.12,.07,.05)*m*cl+vec3(1.,.52,.34)*rim*(.3+cl)*.9,m*.92);
}
vec3 crab(vec2 u){
  vec2 e=mat2(.8,.6,-.6,.8)*u;e.y*=1.45;float r=length(e);vec3 col=vec3(0.);
  if(r<1.4){float n=nz(vec3(e*1.1,.44)),f=nz(vec3(e*2.8,.9));
    col=vec3(.3,.5,1.)*exp(-r*r*2.6)*(.5+.5*n)*.7+vec3(1.,.4,.18)*pow(1.-abs(2.*f-1.),8.)*smoothstep(1.1,.5,r)*(.4+n)*1.4;}
  float pr=length(u),pulse=pow(.5+.5*sin(T*TAU*1.1),30.);
  col+=vec3(.8,.9,1.)*exp(-pr*pr*1200.)*(1.5+8.*pulse);
  vec2 bd=vec2(cos(T*.8),sin(T*.8));float al=dot(u,bd),pe=dot(u,vec2(-bd.y,bd.x)),w=.012+.06*abs(al);
  return col+vec3(.55,.72,1.)*exp(-pe*pe/(w*w))*exp(-abs(al)*2.2)*(.5+1.5*pulse);
}
vec3 spiral(vec2 u){
  u=mat2(.9,.44,-.44,.9)*u;u.y/=.42;float r=length(u);if(r>1.3)return vec3(0.);
  float a=atan(u.y,u.x),sp=2.*(a-log(max(r,.02))*2.4),n=nz(vec3(u*1.4,.63)),disk=exp(-r*3.);
  float arms=pow(.5+.5*cos(sp),3.)*smoothstep(.06,.28,r)*disk*(.45+n);
  float dust=pow(.5+.5*cos(sp+.9),10.)*smoothstep(.1,.3,r)*smoothstep(1.,.35,r);
  float hii=smoothstep(.72,.8,nz(vec3(u*5.,.2)))*arms*3.;
  return (vec3(1.,.84,.58)*(exp(-r*r*60.)*1.8+exp(-r*r*8.)*.3)+vec3(.5,.68,1.)*arms*1.4+vec3(.9,.9,1.)*disk*.1+vec3(1.,.36,.6)*hii)*(1.-.7*dust);
}
vec3 comet(vec3 d,vec3 sd){
  vec3 U=normalize(cross(vec3(0.,1.,0.),CMd)),V=cross(CMd,U);
  vec2 u=vec2(dot(d,U),dot(d,V))/.06,aw=-normalize(vec2(dot(sd,U),dot(sd,V))+1e-5);
  float s=dot(u,aw),p=dot(u,vec2(-aw.y,aw.x)),r2=dot(u,u);
  vec3 col=vec3(.75,.95,1.)*(exp(-r2*400.)*3.+exp(-r2*25.)*.3);
  if(s>0.){
    float ion=exp(-p*p/(.0015+.006*s*s))*exp(-s*.7)*(.55+.45*nz(vec3(s*.8-T*.02,p*6.,.3)));
    float q=p+.12*s*s,dst=exp(-q*q/(.004+.04*s*s))*exp(-s*1.1);
    col+=vec3(.3,.6,1.)*ion+vec3(1.,.82,.52)*dst*.6;
  }
  return col;
}
vec3 sky(vec3 d,vec3 ro){
  vec3 gn=vec3(.3,.94,.15);float bd=dot(d,gn),band=exp(-bd*bd*12.);
  vec3 c=vec3(0.);
  for(int k=0;k<6;k++){
    float fk=float(k),sc=34.+fk*46.;
    vec3 q=d*sc,id=floor(q),f=fract(q)-.5;
    float rnd=hash13(id+fk*31.7);
    if(rnd>.952+fk*.006-band*.04){
      vec3 dd=f-(vec3(hash13(id+1.3),hash13(id+2.9),hash13(id+4.1))-.5)*.6;
      float sz=.022+.026*hash13(id+5.7),fp=PIX*sc*.75,se=sz*sz+fp*fp;
      c+=starCol(hash13(id+8.3))*exp(-dot(dd,dd)/se)*(sz*sz/se)*(.12+8.*pow(hash13(id+6.1),9.));
    }
  }
  {vec3 q=d*22.,id=floor(q),f=fract(q)-.5;
    if(hash13(id+71.3)>.985){
      vec3 e1=normalize(cross(d,vec3(hash13(id+2.),hash13(id+5.),hash13(id+9.))-.5)),e2=cross(d,e1);
      vec3 dd=f-(vec3(hash13(id+11.),hash13(id+13.),hash13(id+17.))-.5)*.5;
      float x=dot(dd,e1),y=dot(dd,e2)/(.25+.6*hash13(id+19.)),s=.018+.03*hash13(id+23.),g=(x*x+y*y)/(s*s);
      c+=mix(vec3(1.,.88,.7),vec3(.72,.84,1.),hash13(id+29.))*exp(-g)*.1*(1.+2.*exp(-g*12.));
    }}
  float neb=smoothstep(.3,.78,nz(d*1.1+nz(d*1.8)*.35)),dust=smoothstep(.44,.62,nz(d*3.4+.31))*band;
  float gc=smoothstep(.5,1.,dot(d,normalize(vec3(-.6,-.1,.79))));
  c+=band*(neb*neb*.9+.015)*(1.+gc*1.5)*mix(vec3(.8,.84,1.),vec3(1.,.8,.55),gc)*.045;
  c*=1.-.65*dust;
  c+=neb*neb*neb*mix(cool,hot,.25)*.006;
  if(dot(d,HXd)>.9929)c+=helix(loc(d,HXd,.085))*.2;
  if(dot(d,PLd)>.971){vec4 pl=pillars(loc(d,PLd,.19));c=c*(1.-pl.a)+pl.rgb*.22;}
  if(dot(d,CRd)>.995)c+=crab(loc(d,CRd,.05))*.22;
  if(dot(d,GXd)>.9928)c+=spiral(loc(d,GXd,.09))*.17;
  // the Sun: corona, prominences, glare, and an occasional coronal mass ejection
  vec3 sv=SUNC-ro;float sD=length(sv);vec3 sdir=sv/sD;float sA=SUNR/sD;
  if(dot(d,CMd)>.966)c+=comet(d,sdir)*.22;
  if(dot(d,sdir)>0.){
    float th=length(cross(d,sdir));float x=th/max(sA,1e-7);
    if(sA<PIX*.6)c+=vec3(1.,.95,.87)*30.*exp(-th*th/(PIX*PIX*.6))*min(1.,sA*sA/(PIX*PIX*.36));
    if(x<60.){
      vec2 u=loc(d,sdir,sA);float r=max(length(u),1.);vec2 cs=u/max(length(u),1e-4);
      c+=vec3(1.,.86,.68)*(exp(-(r-1.)*1.1)*.9+exp(-(r-1.)*.22)*.12)*step(1.,x);
      c+=vec3(1.,.42,.26)*smoothstep(.55,.85,nz(vec3(cs*.9,T*.02)))*exp(-(r-1.25)*(r-1.25)/.03)*5.*step(1.,x);
      float cyc=floor(T/180.),ph=fract(T/180.)*7.,ca=hash13(vec3(cyc,.3,.7))*TAU,spread=1.-dot(cs,vec2(cos(ca),sin(ca)));
      if(ph<1.){float rc=1.2+ph*16.,wid=.35+ph*2.;c+=vec3(1.,.76,.55)*exp(-(r-rc)*(r-rc)/(wid*wid))*exp(-spread*5.)*(1.-ph)*(1.-ph)*(.5+.8*nz(vec3(cs*1.5,r*.1+cyc*.37)))*1.3;}
    }
  }
  // planets too small to show a disc: points of light at their real brightness
  for(int i=0;i<NBOD;i++){
    if(BP[i].x<=0.)continue;
    vec3 pd=BC[i].xyz-ro;float D=length(pd);if(dot(d,pd)<=0.)continue;
    float ang=BC[i].w/D;if(ang>=PIX*.35)continue;
    float th=length(cross(d,pd/D)),sg=PIX*.85;
    c+=PCOL[i]*BP[i].x*exp(-th*th/(sg*sg));
  }
  // the black holes seen from far away: a glow that never shrinks below a few pixels
  vec3 bv=-cam-ro;float bD=length(bv);
  if(CORE>0.&&bD>150.&&dot(d,bv)>0.){
    float th=length(cross(d,bv/bD)),sg=max(24./bD,PIX*3.2),I=(.5+.35*energy+.45*voice)*smoothstep(150.,700.,bD);
    c+=(mix(cool,hot,.72)*.85+ACC*.2)*I*(exp(-th*th/(sg*sg))*1.5+exp(-th*th/(sg*sg*18.))*.1);
  }
  // the AI woven through the sky: a faint web with pulses running along it
  if(NETK>0.){
    float w1=1.-abs(2.*nz(d*1.1+.37)-1.),w2=1.-abs(2.*nz(d*2.3+.71)-1.);
    float web=pow(w1,40.)*.9+pow(w2,56.)*.45,pulse=.35+.65*pow(.5+.5*sin(nz(d*.6+.2)*40.-T*1.5),6.);
    c+=ACC*web*pulse*.026*NETK;
  }
  return c;
}

// ================= what a ray outside the core hits =================
float envHit(vec3 ro,vec3 roW,vec3 rd,float tmax,bool prim,out int id,out vec3 n,out vec3 extra){
  float tb=tmax;id=-1;extra=vec3(0.);n=vec3(0.,1.,0.);
  for(int i=0;i<NBOD;i++){
    if(i==2||BC[i].w<=0.)continue;
    vec3 oc=BC[i].xyz-ro;float D=length(oc);
    if(BC[i].w<D*PIX*.35||dot(oc,rd)<-BC[i].w*1.2)continue;
    vec3 nn;float t=ellHit(ro,rd,i,nn);if(t<tb){tb=t;id=i;n=nn;}
  }
  {vec3 o=prim?camE:ro-BC[2].xyz;float b=dot(o,rd),c=prim?EALT*(EALT+2.*RE):dot(o,o)-RE*RE,h=b*b-c;
    if(h>0.&&b<0.&&c>0.){float t=c/(-b+sqrt(h));if(t<tb){tb=t;id=2;n=normalize(o+rd*t);}}}
  {vec3 sv=SUNC-ro;if(SUNR>length(sv)*PIX*.6){float t=sph(ro,rd,SUNC,SUNR);if(t<tb){tb=t;id=9;n=normalize(ro+rd*t-SUNC);}}}
  if(dot(roW,roW)<4e6){
    vec3 c;float t=asteroids(roW,rd,tb,c);if(t<tb){tb=t;id=10;extra=c;}
    t=hoopHit(roW,rd,tb,c);if(t<tb){tb=t;id=11;extra=c;}
  }
  return tb;
}
vec3 envShade(vec3 ro,vec3 rd,float t,int id,vec3 n,vec3 extra,bool prim){
  vec3 c;
  if(id<0)c=sky(rd,ro);
  else if(id==2)c=earthSurface(ro,rd,t,prim);
  else if(id==3)c=moonS(n,rd,t);
  else if(id==9){float mu=max(dot(n,-rd),0.);float g=fbmL(n*60.,t*PIX/SUNR*60.,6);c=vec3(1.,.93,.84)*30.*(1.-.6*(1.-mu)-.12*(1.-mu)*(1.-mu))*(1.+g*.3);}
  else if(id>=10)c=extra;
  else c=bodyShade(id,n,rd,t);
  float tr;vec4 rc=satRing(ro,rd,tr);if(tr<t)c=c*(1.-rc.a)+rc.rgb;
  return earthAtmos(ro,rd,prim,t,id==2,c);
}

void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/min(R.x,R.y);
  PIX=1./(min(R.x,R.y)*focal);
  vec3 rd0=normalize(view*vec3(uv,focal)),rd=rd0,p=cam;
  float Mt=2.*MH,rh=2.*MH;
  vec3 Lv=cross(p,rd);float h2=dot(Lv,Lv),b=max(sqrt(h2),1e-4),along=dot(p,rd);
  bool outside=dot(cam,cam)>RB*RB;
  float tStop=1e30;vec3 col=vec3(0.);
  // the surface each pixel ends on is shaded once, at the end: one copy of the Earth and sky code keeps the shader small enough
  // for Windows' DirectX compiler (four inlined copies made it slow enough to trip the GPU watchdog)
  vec3 sRo=vec3(0.),sRd=rd0,sN=vec3(0.,1.,0.),sEx=vec3(0.);float sT=1e30,sW=0.;int sId=-1;bool sPrim=true;
  if(outside&&(CORE<.5||b>RB||along>0.)){   // with the core out of view (CORE 0) no ray needs the full march
    // never comes near the black holes: find what it hits, then bend it only by the deflection picked up on the way there
    vec3 rdF=bend(p,rd,Mt*(2.-lensF(along/b))/b);
    int id;vec3 n,ex;float th=envHit(vec3(0.),cam,rdF,1e30,true,id,n,ex);
    vec3 rdS=rdF;
    if(th<1e29){rdS=bend(p,rd,Mt*(lensF((along+th)/b)-lensF(along/b))/b);th=envHit(vec3(0.),cam,rdS,1e30,true,id,n,ex);}
    sRd=rdS;sT=th;sId=id;sN=n;sEx=ex;sW=1.;
    tStop=th;
  }else{
    float trav=outside?max(-along-sqrt(max(RB*RB-h2,0.)),0.):0.;
    bool done=false;
    if(trav>0.){int id;vec3 n,ex;float th=envHit(vec3(0.),cam,rd,trav,true,id,n,ex);if(th<trav){sRd=rd;sT=th;sId=id;sN=n;sEx=ex;sW=1.;tStop=th;done=true;}}
    if(!done){
      if(trav>0.)rd=bend(p,rd,Mt*(lensF((along+trav)/b)-lensF(along/b))/b);
      p+=rd*trav;
      float dith=ign(gl_FragCoord.xy+frame*5.588238);
      float tr=1.,m1=1e3,m2=1e3;bool fell=false,wall=false;
      float bd[NL],bs[NL],bt[NL],bx[NL],hd[NL],ht[NL],hx[NL];
      for(int k=0;k<NL;k++){bd[k]=.1;hd[k]=.1;bs[k]=0.;bt[k]=0.;bx[k]=0.;ht[k]=0.;hx[k]=0.;}
      float cBPH=cos(BPH),sBPH=sin(BPH);
      for(int i=0;i<STEPS;i++){
        vec3 d1=p-H1,d2=p-H2;float r1=length(d1),r2=length(d2),r=length(p);
        m1=min(m1,r1);m2=min(m2,r2);
        if(r1<rh||r2<rh){fell=true;break;}
        if(r>RB+.6&&dot(p,rd)>0.)break;
        float rn=min(r1,r2),dt=.05*rn+.02,rxz=length(p.xz);
        if(rxz>CBIN*.75&&rxz<CB1){float H=.05+.03*rxz;dt=min(dt,max(.6*H,(abs(p.y)-2.5*H)*.7));}
        if(rn<MD1+.4){float Hm=.02+.03*rn;dt=min(dt,max(.6*Hm,(abs(p.y)-2.5*Hm)*.7));}
        vec2 lq,lg;float dl=lem(p,lq,lg);if(rxz<LEMA+1.5)dt=min(dt,max(.035,(max(dl,abs(p.y))-.32)*.7));
        if(i==0)dt*=dith+.02;
        // light bending by both holes (superposed Schwarzschild photon orbits), midpoint step
        vec3 L1=cross(d1,rd),L2=cross(d2,rd);
        vec3 acc=-3.*MH*(dot(L1,L1)*d1/pow(r1,5.)+dot(L2,L2)*d2/pow(r2,5.));
        vec3 rdm=normalize(rd+acc*dt*.5),pm=p+rd*dt*.5;
        vec3 e1=pm-H1,e2=pm-H2;float q1=max(length(e1),rh*.9),q2=max(length(e2),rh*.9);vec3 M1=cross(e1,rdm),M2=cross(e2,rdm);
        vec3 np=p+rdm*dt;rd=normalize(rd-3.*MH*(dot(M1,M1)*e1/pow(q1,5.)+dot(M2,M2)*e2/pow(q2,5.))*dt);
        // the lights (the AI's drones), exact closest approach so the tails stay smooth
        vec3 sm=(p+np)*.5;float sh=dt*.5,s,u,dd2;
        for(int k=0;k<NL;k++){
          vec4 Bk=LB[k];vec3 dc=sm-Bk.xyz;float lim=Bk.w+sh;
          if(dot(dc,dc)>lim*lim)continue;
          dd2=ptSeg(LP[k*NP].xyz,p,np,s);
          if(dd2<hd[k]){hd[k]=dd2;ht[k]=tr;hx[k]=trav+s*dt;}
          for(int j=0;j<NP-1;j++){
            vec4 Pa=LP[k*NP+j];vec3 Pb=LP[k*NP+j+1].xyz,mc=sm-(Pa.xyz+Pb)*.5;float l2=Pa.w+sh+.32;
            if(dot(mc,mc)>l2*l2)continue;
            dd2=segSeg(p,np,Pa.xyz,Pb,s,u);
            if(dd2<bd[k]){bd[k]=dd2;bs[k]=(float(j)+u)/float(NP-1);bt[k]=tr;bx[k]=trav+s*dt;}
          }
        }
        // the collector ring: found by the sign change across its radius
        float f0=length(p.xz)-RINGR,f1=length(np.xz)-RINGR;
        if(f0*f1<0.){float s2=f0/(f0-f1);vec3 hp=mix(p,np,s2);
          if(abs(hp.y)<RINGH){float al;vec3 rc=ringShade(hp,rdm,al);col+=tr*rc*al;tr*=1.-al;if(tr<.01){wall=true;tStop=trav+s2*dt;break;}}}
        float boost=(1.+energy*.6+surge*.7);
        // 1) circumbinary disk: inner edge at two separations, two spiral arms and an overdense lump driven by the binary
        float rr=length(pm.xz);
        if(rr>CBIN*.8&&rr<CB1){float H=.05+.03*rr;if(abs(pm.y)<2.5*H){
          float x=sqrt(CBIN/rr),x2=x*x,F=max(x2*x2*x2*(1.-x),0.)*17.6,ang=atan(pm.z,pm.x);
          float spiral=.5+.5*cos(2.*(ang-BPH)-3.2*log(rr/CBIN)),lump=1.+1.3*exp(-(rr-CBIN*1.2)*(rr-CBIN*1.2)*.4)*pow(.5+.5*cos(ang-BPH*.23),6.);
          float gw=1.+.07*cos(2.*(ang-BPH)+rr*.9-T*.0);   // gravitational-wave ripple, two-armed like the pattern the binary radiates
          float dens=gas(rr,ang,pm.y,1.7*pow(3./rr,1.5),0.)*exp(-pm.y*pm.y/(2.*H*H))*smoothstep(CB1,CB1-6.,rr)*smoothstep(CBIN*.8,CBIN,rr)*(.4+.8*spiral)*lump*gw;
          vec3 vel=normalize(vec3(-pm.z,0.,pm.x));float beta=clamp(sqrt(Mt/rr),0.,.7);
          float g=sqrt(1.-beta*beta)/(1.-beta*dot(vel,-rdm))*sqrt(max(0.,1.-rh*2./rr));
          float To=pow(F,.25)*g*.8,em=dens*F*g*g*g*.36*(1.+voice*(.45+.75*smoothstep(-.3,1.,sin(rr*1.3-T*5.))))*boost;
          col+=tr*thermal(To,g)*em*dt*5.;tr*=exp(-dens*dt*1.6);
        }}
        // 2) the two mini-disks, hottest just outside each hole
        for(int k=0;k<2;k++){
          vec3 hc=k==0?H1:H2,dm=pm-hc;float ri=length(dm.xz);
          if(ri>MD0*.9&&ri<MD1){float Hm=.02+.03*ri;if(abs(dm.y)<2.5*Hm){
            float x=sqrt(MD0/ri),x2=x*x,F=max(x2*x2*x2*(1.-x),0.)*17.6*1.5;
            float dens=gas(ri*3.,atan(dm.z,dm.x),dm.y*3.,2.6*pow(1.2/ri,1.5),5.+float(k)*3.)*exp(-dm.y*dm.y/(2.*Hm*Hm))*smoothstep(MD1,MD1-.35,ri)*smoothstep(MD0*.9,MD0,ri);
            vec3 vel=normalize(vec3(-dm.z,0.,dm.x));float beta=clamp(sqrt(MH/ri),0.,.75);
            float g=sqrt(1.-beta*beta)/(1.-beta*dot(vel,-rdm))*sqrt(max(0.,1.-rh/ri));
            col+=tr*thermal(pow(F,.25)*g*1.05,g)*dens*F*g*g*g*2.1*boost*dt*5.;tr*=exp(-dens*dt*2.);
          }}
        }
        // 3) the figure-eight stream: gas sloshing between the holes through the point between them
        vec2 mq,mg;float dcur=lem(pm,mq,mg),d2s=dcur*dcur+pm.y*pm.y*4.;
        if(d2s<.5){
          float w=.15,ang=atan(mq.y,mq.x),near=min(length(pm-H1),length(pm-H2));
          float heat=exp(-near*near/2.5)*1.6+.4,flow=.5+.5*nz(vec3(ang*1.3-DT*.9,mq*.4));
          float dens=exp(-d2s/(w*w))*flow*heat;
          vec2 tg=normalize(vec2(-mg.y,mg.x)+1e-6)*sign(mq.x+1e-4);vec3 vel=vec3(cBPH*tg.x-sBPH*tg.y,0.,sBPH*tg.x+cBPH*tg.y);
          float beta=.35,g=sqrt(1.-beta*beta)/(1.-beta*dot(vel,-rdm));
          col+=tr*thermal(.95*g,g)*dens*g*g*g*3.4*boost*dt;tr*=exp(-dens*dt*.7);
        }
        if(tr<.008)break;
        trav+=dt;p=np;
      }
      if(!fell&&!wall&&tr>.01){
        vec3 L2=cross(p,rd);float b2=max(length(L2),1e-3);vec3 rdo=bend(p,rd,Mt*(2.-lensF(dot(p,rd)/b2))/b2);
        int id;vec3 n,ex;float th=envHit(p-cam,p,rdo,1e30,false,id,n,ex);
        sRo=p-cam;sRd=rdo;sT=th;sId=id;sN=n;sEx=ex;sPrim=false;sW=tr;
      }
      if(fell)tStop=max(-dot(cam,rd0),0.);
      // photon rings round each hole
      float q1=(m1/rh-1.5)*16.,q2=(m2/rh-1.5)*16.;
      col+=hot*(exp(-q1*q1)+exp(-q2*q2))*(.3+energy*.5+voice*1.+surge*1.4)*.55*tr;
      // the drones: a white-hot head and a thin rainbow tail
      for(int k=0;k<NL;k++){
        vec4 C=LC[k];
        if(bd[k]<.1){
          float s=bs[k],fp=bx[k]*PIX,w=mix(.024,.009,s),we2=w*w+fp*fp*.6;
          vec3 prof=exp(-bd[k]/(we2*vec3(1.5,1.,.66)))*w*inversesqrt(we2)+exp(-bd[k]*70.)*.05;
          vec3 rgb=mix(mix(rainbow(fract(C.x+T*.08+s*.45)),vec3(1.),.55*exp(-s*10.)),vec3(1.),mono);
          col+=bt[k]*rgb*prof*(1.-s)*(1.-s)*C.y*trailGlow;
        }
        if(hd[k]<.1){
          float fp=hx[k]*PIX,w2=.03*.03+fp*fp*.6;
          col+=ht[k]*mix(mix(rainbow(fract(C.x+T*.08)),vec3(1.),.6),vec3(1.),mono)*(exp(-hd[k]/w2)*.03*inversesqrt(w2)*2.+exp(-hd[k]*45.)*.12)*C.y*trailGlow;
        }
      }
    }
  }
  if(sW>0.)col+=sW*envShade(sRo,sRd,sT,sId,sN,sEx,sPrim);
  // light that isn't a surface, along the straight primary ray: the jets and the AI's filaments
  if(CORE>0.)col+=jets(cam,rd0,tStop);
  if(NETK*FIL>0.)col+=filaments(cam,rd0,tStop)*NETK;
  o=vec4(col,1.);
}`;

const RESOLVE=`#version 300 es
precision highp float;in vec2 v;out vec4 o;uniform sampler2D S;uniform vec2 outSize,scale,texSize;uniform float ca,sharp;
vec3 tap(vec2 uv){vec2 c=uv-.5,lim=scale-.5/texSize;float k=dot(c,c)*ca;
  return vec3(texture(S,clamp((uv+c*k)*scale,vec2(0.),lim)).r,texture(S,clamp(uv*scale,vec2(0.),lim)).g,texture(S,clamp((uv-c*k)*scale,vec2(0.),lim)).b);}
void main(){vec2 px=1./outSize;
  vec3 a=(tap(v+vec2(.125,.375)*px)+tap(v+vec2(-.375,.125)*px)+tap(v+vec2(.375,-.125)*px)+tap(v+vec2(-.125,-.375)*px))*.25;
  if(sharp>0.){vec2 e=1./(texSize*scale);
    vec3 bl=(tap(v+vec2(e.x,0.))+tap(v-vec2(e.x,0.))+tap(v+vec2(0.,e.y))+tap(v-vec2(0.,e.y)))*.25;
    a=max(a+sharp*(a-bl),a*.6);}
  o=vec4(a,1.);}`;
const DOWN=`#version 300 es
precision highp float;in vec2 v;out vec4 o;uniform sampler2D S;uniform vec2 texel;uniform int first;
vec3 t(vec2 f){return texture(S,v+f*texel).rgb;}
float lum(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
vec3 kw(vec3 c){return c/(1.+lum(c));}
void main(){
  vec3 a=t(vec2(-2,2)),b=t(vec2(0,2)),c=t(vec2(2,2)),d=t(vec2(-2,0)),e=t(vec2(0,0)),f=t(vec2(2,0)),g=t(vec2(-2,-2)),h=t(vec2(0,-2)),i=t(vec2(2,-2)),j=t(vec2(-1,1)),k=t(vec2(1,1)),l=t(vec2(-1,-1)),m=t(vec2(1,-1));
  vec3 res;
  if(first==1){res=kw((j+k+l+m)*.25)*.5+(kw((a+b+d+e)*.25)+kw((b+c+e+f)*.25)+kw((d+e+g+h)*.25)+kw((e+f+h+i)*.25))*.125;res=res/(1.-min(lum(res),.999));
    float L=lum(res),x=max(L-.05,0.);res*=x*x/(x+.03)/max(L,1e-4);}
  else res=e*.125+(a+c+g+i)*.03125+(b+d+f+h)*.0625+(j+k+l+m)*.125;
  o=vec4(res,1.);}`;
const UP=`#version 300 es
precision highp float;in vec2 v;out vec4 o;uniform sampler2D S;uniform vec2 texel;
void main(){vec2 d=texel;
  vec3 r=texture(S,v+vec2(-d.x,d.y)).rgb+2.*texture(S,v+vec2(0.,d.y)).rgb+texture(S,v+d).rgb
        +2.*texture(S,v+vec2(-d.x,0.)).rgb+4.*texture(S,v).rgb+2.*texture(S,v+vec2(d.x,0.)).rgb
        +texture(S,v-d).rgb+2.*texture(S,v+vec2(0.,-d.y)).rgb+texture(S,v+vec2(d.x,-d.y)).rgb;
  o=vec4(r/16.,1.);}`;
const STREAK=`#version 300 es
precision highp float;in vec2 v;out vec4 o;uniform sampler2D S;uniform vec2 texel;uniform float thresh;
void main(){vec3 acc=vec3(0.);float ws=0.;
  for(int i=-24;i<=24;i++){float fi=float(i),w=exp(-abs(fi)*.11);acc+=max(texture(S,v+vec2(fi*texel.x*1.25,0.)).rgb-vec3(thresh),0.)*w;ws+=w;}
  o=vec4(acc/ws,1.);}`;
const COMP=`#version 300 es
precision highp float;in vec2 v;out vec4 o;
uniform sampler2D HDRI,BLOOM,STRK;uniform vec2 R;uniform float T,bloomK,streakK,exposure,cine,mono;uniform vec3 tint;
vec3 aces(vec3 c){
  const mat3 AIN=mat3(.59719,.07600,.02840,.35458,.90834,.13383,.04823,.01566,.83777);
  const mat3 AOUT=mat3(1.60475,-.10208,-.00327,-.53108,1.10813,-.07276,-.07367,-.00605,1.07602);
  c=AIN*c;vec3 a=c*(c+.0245786)-.000090537,b=c*(.983729*c+.4329510)+.238081;return clamp(AOUT*(a/b),0.,1.);}
float h(vec2 p){vec3 q=fract(vec3(p.xyx)*.1031);q+=dot(q,q.yzx+33.33);return fract((q.x+q.y)*q.z);}
float lum(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
void main(){
  vec3 col=texture(HDRI,v).rgb+texture(BLOOM,v).rgb*bloomK+texture(STRK,v).rgb*tint*streakK;
  col=max(mix(vec3(lum(col)),col,1.15),0.);
  if(mono>.5)col=vec3(lum(col));
  col=pow(aces(col*exposure),vec3(1./2.2));
  if(mono>.5)col=mix(col,smoothstep(0.,1.,col),.35);
  else{float l=lum(col),s=max(col.r,max(col.g,col.b))-min(col.r,min(col.g,col.b));col=mix(vec3(l),col,1.+.3*(1.-s));col=pow(max(col,0.),vec3(1.02,1.,.97));}
  col=col*col/(col+.03)*1.03;
  vec2 c=v-.5;float asp=R.x/R.y;
  col*=mix(1.,smoothstep(1.2,.28,length(c*vec2(asp*.7,1.))),.6);
  col+=(h(v*R+fract(T*7.)*113.)-.5)*.03*sqrt(max(col,0.))*(1.-col*.6);
  float bar=(1.-asp/2.39)*.5*cine;
  if(bar>0.&&(v.y<bar||v.y>1.-bar))col=vec3(0.);
  o=vec4(clamp(col,0.,1.),1.);}`;
// reprojects a mosaic of Web-Mercator satellite tiles into a local tangent-plane window (mesh carries exact source coordinates)
const REPROJ_VS=`#version 300 es
in vec2 a;in vec2 s;out vec2 vs;void main(){vs=s;gl_Position=vec4(a,0.,1.);}`;
const REPROJ_FS=`#version 300 es
precision highp float;in vec2 vs;out vec4 o;uniform sampler2D STG;void main(){o=texture(STG,vs);}`;
// orbit lines, drawn over the finished frame
const LINE_VS=`#version 300 es
in vec3 a;uniform mat3 VW;uniform vec2 SC;uniform float F;void main(){vec3 q=VW*a;gl_Position=vec4(q.x*F*SC.x,q.y*F*SC.y,q.z*.5,q.z);}`;
const LINE_FS=`#version 300 es
precision highp float;uniform vec4 C;out vec4 o;void main(){o=C;}`;

(function(){
const $=id=>document.getElementById(id);
const cv=$("c");
const gl=cv.getContext("webgl2",{antialias:false,alpha:false,depth:false,stencil:false,powerPreference:"high-performance"});
if(!gl){$("ro1").textContent="WebGL 2 isn't available in this browser";return;}
const HDR=!!gl.getExtension("EXT_color_buffer_float");
gl.getExtension("OES_texture_float_linear");
const TQ=gl.getExtension("EXT_disjoint_timer_query_webgl2");
const ANI=gl.getExtension("EXT_texture_filter_anisotropic");
const IFMT=HDR?gl.RGBA16F:gl.RGBA8,TYPE=HDR?gl.HALF_FLOAT:gl.UNSIGNED_BYTE;
let GPU="";
try{const d=gl.getExtension("WEBGL_debug_renderer_info");if(d){const s=String(gl.getParameter(d.UNMASKED_RENDERER_WEBGL));const m=s.match(/ANGLE \([^,]*,\s*([^,(]+)/);GPU=(m?m[1]:s).trim();}}catch(e){}
const SOFTWARE=/swiftshader|llvmpipe|softpipe|basic render|software/i.test(GPU),INTEGRATED=SOFTWARE||/intel|uhd|iris|radeon\(tm\) graphics|vega/i.test(GPU),DISCRETE=/nvidia|geforce|rtx|gtx|radeon rx|arc a/i.test(GPU);

// ---------- vector helpers (double precision) ----------
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]],sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],mul=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const len=a=>Math.hypot(a[0],a[1],a[2]),norm=a=>{const l=len(a)||1;return[a[0]/l,a[1]/l,a[2]/l];};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x)),TWO_PI=2*Math.PI;
function rotv(v,k,a){const c=Math.cos(a),s=Math.sin(a),d=dot(k,v)*(1-c),x=cross(k,v);return[v[0]*c+x[0]*s+k[0]*d,v[1]*c+x[1]*s+k[1]*d,v[2]*c+x[2]*s+k[2]*d];}
function slerp(a,b,t){const d=clamp(dot(a,b),-1,1),w=Math.acos(d);if(w<1e-6)return b.slice();
  if(Math.PI-w<1e-3){let k=cross(a,[0,1,0]);if(len(k)<1e-6)k=cross(a,[1,0,0]);return rotv(a,norm(k),w*t);}
  const s=Math.sin(w);return norm(add(mul(a,Math.sin((1-t)*w)/s),mul(b,Math.sin(t*w)/s)));}
const mapply=(M,v)=>[M[0][0]*v[0]+M[1][0]*v[1]+M[2][0]*v[2],M[0][1]*v[0]+M[1][1]*v[1]+M[2][1]*v[2],M[0][2]*v[0]+M[1][2]*v[1]+M[2][2]*v[2]]; // astronomy-engine layout
const qApply=(Q,v)=>add(add(mul(Q[0],v[0]),mul(Q[1],v[1])),mul(Q[2],v[2]));            // Q = columns (images of the frame's axes)
const qInv=(Q,v)=>[dot(Q[0],v),dot(Q[1],v),dot(Q[2],v)];

// ---------- tileable 3D noise, baked once ----------
function bakeNoise(N){
  let seed=1337;const rnd=()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};
  const oct=[[8,.5],[16,.3],[32,.2]].map(([p,a])=>{const l=new Float32Array(p*p*p);for(let i=0;i<l.length;i++)l[i]=rnd();return{p,a,l};});
  const out=new Uint8Array(N*N*N),q=t=>t*t*t*(t*(t*6-15)+10);let i=0;
  for(let z=0;z<N;z++)for(let y=0;y<N;y++)for(let x=0;x<N;x++){
    let s=0;
    for(const o of oct){
      const fx=x/N*o.p,fy=y/N*o.p,fz=z/N*o.p,ix=fx|0,iy=fy|0,iz=fz|0,tx=q(fx-ix),ty=q(fy-iy),tz=q(fz-iz),p=o.p,l=o.l;
      const x1=(ix+1)%p,y1=(iy+1)%p,z1=(iz+1)%p,Z0=iz*p*p,Z1=z1*p*p,Y0=iy*p,Y1=y1*p;
      const a=l[Z0+Y0+ix]+(l[Z0+Y0+x1]-l[Z0+Y0+ix])*tx,b=l[Z0+Y1+ix]+(l[Z0+Y1+x1]-l[Z0+Y1+ix])*tx;
      const c=l[Z1+Y0+ix]+(l[Z1+Y0+x1]-l[Z1+Y0+ix])*tx,d=l[Z1+Y1+ix]+(l[Z1+Y1+x1]-l[Z1+Y1+ix])*tx;
      const e=a+(b-a)*ty,f=c+(d-c)*ty;s+=o.a*(e+(f-e)*tz);
    }
    out[i++]=Math.max(0,Math.min(255,Math.round(s*255)));
  }
  return out;
}
const NN=64,noise=gl.createTexture();
gl.bindTexture(gl.TEXTURE_3D,noise);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
gl.texImage3D(gl.TEXTURE_3D,0,gl.R8,NN,NN,NN,0,gl.RED,gl.UNSIGNED_BYTE,bakeNoise(NN));
for(const k of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T,gl.TEXTURE_WRAP_R])gl.texParameteri(gl.TEXTURE_3D,k,gl.REPEAT);
gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);

// ---------- programs ----------
// compiled in the background where the browser allows it (KHR_parallel_shader_compile), so the page never freezes while it starts
const PCX=gl.getExtension("KHR_parallel_shader_compile");
function program(vs,fs){
  const p=gl.createProgram(),sh=[];
  for(const [type,src] of [[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);gl.attachShader(p,s);sh.push(s);}
  gl.bindAttribLocation(p,0,"p");gl.bindAttribLocation(p,0,"a");gl.bindAttribLocation(p,1,"s");gl.linkProgram(p);
  const cache={};return{p,sh,u:n=>(n in cache)?cache[n]:(cache[n]=gl.getUniformLocation(p,n))};
}
function programsReady(){return !PCX||Object.values(P).every(x=>gl.getProgramParameter(x.p,PCX.COMPLETION_STATUS_KHR));}
function programsOK(){let ok=true;for(const x of Object.values(P)){if(!gl.getProgramParameter(x.p,gl.LINK_STATUS)){ok=false;x.sh.forEach(s=>{const l=gl.getShaderInfoLog(s);if(l)console.error(l);});console.error(gl.getProgramInfoLog(x.p));}}return ok;}
const P={scene:program(VS,SCENE),resolve:program(VS,RESOLVE),down:program(VS,DOWN),up:program(VS,UP),streak:program(VS,STREAK),comp:program(VS,COMP),reproj:program(REPROJ_VS,REPROJ_FS),line:program(LINE_VS,LINE_FS)};
const vao=gl.createVertexArray();gl.bindVertexArray(vao);
gl.bindBuffer(gl.ARRAY_BUFFER,gl.createBuffer());gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);

function target(w,h){const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D,0,IFMT,w,h,0,gl.RGBA,TYPE,null);
  const f=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,f);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,t,0);
  return{t,f,w,h};}
function free(x){if(x){gl.deleteTexture(x.t);gl.deleteFramebuffer(x.f);}}
const MAXTEX=gl.getParameter(gl.MAX_TEXTURE_SIZE);
let sceneT=null,hdrT=null,mips=[],streakT=null,W=0,H=0;
function ensure(w,h){
  if(w===W&&h===H)return;W=w;H=h;
  [sceneT,hdrT,streakT,...mips].forEach(free);mips=[];
  const k=Math.min(2,Math.sqrt(16.6e6/(W*H)));
  sceneT=target(Math.min(MAXTEX,Math.round(W*k)),Math.min(MAXTEX,Math.round(H*k)));
  hdrT=target(W,H);
  let a=W,b=H;for(let i=0;i<6;i++){a=Math.max(1,a>>1);b=Math.max(1,b>>1);mips.push(target(a,b));}
  streakT=target(mips[1].w,mips[1].h);
}
function pass(prog,dst,fn,vw,vh){gl.useProgram(prog.p);gl.bindFramebuffer(gl.FRAMEBUFFER,dst?dst.f:null);gl.viewport(0,0,vw||(dst?dst.w:W),vh||(dst?dst.h:H));fn(prog.u);gl.bindVertexArray(vao);gl.drawArrays(gl.TRIANGLES,0,3);}
function tex(u,name,unit,t,kind){gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(kind||gl.TEXTURE_2D,t);gl.uniform1i(u(name),unit);}

// ---------- Earth textures: NASA imagery bundled with the page, sharpened by live sources where the network allows ----------
function mkTex(w,h,wrapS,fmt,ifmt){const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);
  gl.texImage2D(gl.TEXTURE_2D,0,ifmt||gl.RGBA8,w,h,0,fmt||gl.RGBA,gl.UNSIGNED_BYTE,null);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,wrapS);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  if(ANI)gl.texParameterf(gl.TEXTURE_2D,ANI.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(8,gl.getParameter(ANI.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
  return t;}
function loadImg(src,cors){return new Promise(res=>{const im=new Image();if(cors)im.crossOrigin="anonymous";im.decoding="async";im.onload=()=>res(im);im.onerror=()=>res(null);im.src=src;});}
function upload(t,im,fmt,ifmt){gl.bindTexture(gl.TEXTURE_2D,t);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);gl.texImage2D(gl.TEXTURE_2D,0,ifmt||gl.RGBA8,fmt||gl.RGBA,gl.UNSIGNED_BYTE,im);gl.generateMipmap(gl.TEXTURE_2D);}
const EGT=mkTex(4,4,gl.REPEAT),ENT=mkTex(4,4,gl.REPEAT,gl.RED,gl.R8),EWT=mkTex(4,4,gl.REPEAT,gl.RED,gl.R8),ECT=mkTex(4,4,gl.REPEAT,gl.RED,gl.R8),ETT=mkTex(4,4,gl.REPEAT,gl.RED,gl.R8);
let GTEX=TWO_PI*13/5400,CTEX=TWO_PI*13/4096,cloudLive=false,cloudStamp=0;
const IMG=window.__EARTH||{};
loadImg(IMG.day,true).then(im=>{if(im){upload(EGT,im);GTEX=TWO_PI*13/im.width;}});
__prepEarth(IMG.lights,"lights").then(im=>{if(im)upload(ENT,im,gl.RED,gl.R8);});
loadImg(IMG.water,true).then(im=>{if(im)upload(EWT,im,gl.RED,gl.R8);});
loadImg(IMG.topo,true).then(im=>{if(im)upload(ETT,im,gl.RED,gl.R8);});
__prepEarth(IMG.clouds,"clouds").then(im=>{if(im&&!cloudLive){upload(ECT,im,gl.RED,gl.R8);CTEX=TWO_PI*13/im.width;}});
// live clouds from EUMETSAT geostationary satellites (via clouds.matteason.co.uk), refreshed every 3 hours
function liveClouds(){
  const size=MAXTEX>=8192&&!INTEGRATED?"8192x4096":"4096x2048";
  loadImg(`https://clouds.matteason.co.uk/images/${size}/clouds.jpg?t=${Math.floor(Date.now()/108e5)}`,true).then(im=>{
    if(!im)return;try{upload(ECT,im,gl.RED,gl.R8);CTEX=TWO_PI*13/im.width;cloudLive=true;cloudStamp=Date.now();credits();}catch(e){}});
}
liveClouds();setInterval(liveClouds,108e5);

// ---------- Esri World Imagery, streamed round wherever you look, down to about 0.3 m per pixel ----------
// Four nested windows (a clipmap). Each is a 2048² tangent-plane image: Web-Mercator tiles are reprojected into it on the GPU,
// and the scene shader reads it with camera-relative offsets computed here in double precision (no float jitter up close).
const SATN=INTEGRATED?1024:2048,SATL=4,ESRI=(z,y,x)=>`https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`;
let satTex=null,stageTex=null,satFB=null,stageFB=null,esriState="trying",esriFails=0,esriOK=0;
const WIN=[0,1,2,3].map(k=>({k,front:k*2,back:k*2+1,live:false,C:[1,0,0],E:[0,0,1],N:[0,1,0],S:1,texel:0,key:"",pend:"",busy:false}));
function satInit(){
  if(satTex)return;
  satTex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D_ARRAY,satTex);
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY,Math.log2(SATN)+1,gl.RGBA8,SATN,SATN,SATL*2);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  if(ANI)gl.texParameterf(gl.TEXTURE_2D_ARRAY,ANI.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(8,gl.getParameter(ANI.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
  stageTex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,stageTex);gl.texStorage2D(gl.TEXTURE_2D,1,gl.RGBA8,2816,2816);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  satFB=gl.createFramebuffer();stageFB=gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER,stageFB);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,stageTex,0);
}
const G=64,meshVAO=gl.createVertexArray(),meshBuf=gl.createBuffer(),meshIdx=gl.createBuffer();
{gl.bindVertexArray(meshVAO);gl.bindBuffer(gl.ARRAY_BUFFER,meshBuf);gl.bufferData(gl.ARRAY_BUFFER,(G+1)*(G+1)*16,gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,16,0);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,2,gl.FLOAT,false,16,8);
  const idx=new Uint16Array(G*G*6);let n=0;for(let j=0;j<G;j++)for(let i=0;i<G;i++){const a=j*(G+1)+i,b=a+1,c=a+G+1,d=c+1;idx.set([a,b,d,a,d,c],n);n+=6;}
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,meshIdx);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx,gl.STATIC_DRAW);gl.bindVertexArray(vao);}
const imgCache=new Map(),queue=[];let inflight=0;
function fetchImg(url){
  if(imgCache.has(url)){const p=imgCache.get(url);imgCache.delete(url);imgCache.set(url,p);return p;}
  const p=new Promise(res=>{queue.push({url,res});pump();});
  imgCache.set(url,p);if(imgCache.size>700)imgCache.delete(imgCache.keys().next().value);return p;
}
function pump(){while(inflight<12&&queue.length){const {url,res}=queue.pop();inflight++;const im=new Image();im.crossOrigin="anonymous";
  im.onload=()=>{inflight--;esriOK++;res(im);pump();};im.onerror=()=>{inflight--;esriFails++;res(null);pump();};im.src=url;}}
const pcv=document.createElement("canvas");pcv.width=pcv.height=16;const pcx=pcv.getContext("2d",{willReadFrequently:true});
function placeholder(im){   // Esri serves a flat grey "Map data not yet available" tile where it has no imagery at that zoom
  try{pcx.drawImage(im,0,0,16,16);const d=pcx.getImageData(0,0,16,16).data;let flat=0;
    for(let i=0;i<d.length;i+=4){const r=d[i],g=d[i+1],b=d[i+2];if(Math.abs(r-g)<5&&Math.abs(g-b)<5&&r>150&&r<242)flat++;}return flat>=246;}catch(e){return true;}
}
const RM=6371000,merc=(lat,z)=>(1-Math.log(Math.tan(Math.PI/4+lat/2))/Math.PI)/2*256*2**z;
function efDir(lat,lon){return[Math.cos(lat)*Math.cos(lon),Math.sin(lat),-Math.cos(lat)*Math.sin(lon)];}   // Earth-fixed: y north, x Greenwich, z 90°W
function efLatLon(q){return[Math.asin(clamp(q[1],-1,1)),Math.atan2(-q[2],q[0])];}
// Esri has zoom 20 (~0.15 m) over many cities; where it doesn't, remember that for the area and use 19
const zCap=new Map(),capKey=(la,lo)=>`${Math.round(la*5730)}/${Math.round(lo*5730)}`;
async function buildWin(w,c,texelM,zmax){
  const [lat0,lon0]=efLatLon(c);
  let z=clamp(Math.round(Math.log2(156543.034*Math.cos(lat0)/texelM)),2,zmax||zCap.get(capKey(lat0,lon0))||20);
  const E=norm(len(cross([0,1,0],c))>1e-6?cross([0,1,0],c):[0,0,-1]),N=cross(c,E),Sm=SATN*texelM/2,S=Sm/RM*13;
  const key=`${z}/${c.map(x=>x.toFixed(6))}/${texelM}`;if(key===w.key||key===w.pend)return;
  w.pend=key;
  // exact source coordinates for every vertex of a 64×64 mesh over the window
  let pts,x0,y0,tx0,ty0,nx,ny;
  for(;;){
    pts=new Float64Array((G+1)*(G+1)*2);const wpx=256*2**z,xc=(lon0/TWO_PI+.5)*wpx;let mnx=1e18,mxx=-1e18,mny=1e18,mxy=-1e18;
    for(let j=0;j<=G;j++)for(let i=0;i<=G;i++){
      const e=(2*i/G-1)*Sm,n=(2*j/G-1)*Sm,h=Math.sqrt(Math.max(0,RM*RM-e*e-n*n));
      const P=add(add(mul(c,h),mul(E,e)),mul(N,n)),[la,lo]=efLatLon(norm(P));
      let x=(lo/TWO_PI+.5)*wpx;x+=Math.round((xc-x)/wpx)*wpx;const y=merc(clamp(la,-1.4844,1.4844),z);
      const o=(j*(G+1)+i)*2;pts[o]=x;pts[o+1]=y;mnx=Math.min(mnx,x);mxx=Math.max(mxx,x);mny=Math.min(mny,y);mxy=Math.max(mxy,y);
    }
    tx0=Math.floor(mnx/256);ty0=Math.max(0,Math.floor(mny/256));nx=Math.floor(mxx/256)-tx0+1;ny=Math.min(2**z-1,Math.floor(mxy/256))-ty0+1;
    if(nx<=11&&ny<=11)break;z--;if(z<2)return;
  }
  satInit();
  gl.bindFramebuffer(gl.FRAMEBUFFER,stageFB);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
  const n2=2**z,jobs=[];let good=0;
  for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){const X=((tx0+i)%n2+n2)%n2,Y=ty0+j;
    jobs.push(fetchImg(ESRI(z,Y,X)).then(im=>{if(!im||w.pend!==key||placeholder(im))return;good++;gl.bindTexture(gl.TEXTURE_2D,stageTex);gl.pixelStorei(gl.UNPACK_ALIGNMENT,4);gl.texSubImage2D(gl.TEXTURE_2D,0,i*256,j*256,gl.RGBA,gl.UNSIGNED_BYTE,im);}));}
  await Promise.all(jobs);
  if(w.pend!==key)return;
  if(z>19&&good<jobs.length*.7){zCap.set(capKey(lat0,lon0),z-1);w.pend="";return buildWin(w,c,texelM,z-1);}
  if(esriOK===0){w.pend="";if(esriFails>=4){esriState="blocked";credits();}return;}
  esriState="ok";
  // reproject into the back layer, then swap it in
  const vb=new Float32Array((G+1)*(G+1)*4);
  for(let j=0;j<=G;j++)for(let i=0;i<=G;i++){const o=j*(G+1)+i;vb[o*4]=2*i/G-1;vb[o*4+1]=2*j/G-1;vb[o*4+2]=(pts[o*2]-tx0*256)/2816;vb[o*4+3]=(pts[o*2+1]-ty0*256)/2816;}
  gl.bindFramebuffer(gl.FRAMEBUFFER,satFB);gl.framebufferTextureLayer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,satTex,0,w.back);
  gl.viewport(0,0,SATN,SATN);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(P.reproj.p);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,stageTex);gl.uniform1i(P.reproj.u("STG"),0);
  gl.bindVertexArray(meshVAO);gl.bindBuffer(gl.ARRAY_BUFFER,meshBuf);gl.bufferSubData(gl.ARRAY_BUFFER,0,vb);
  gl.disable(gl.BLEND);gl.drawElements(gl.TRIANGLES,G*G*6,gl.UNSIGNED_SHORT,0);gl.bindVertexArray(vao);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY,satTex);gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
  [w.front,w.back]=[w.back,w.front];w.C=c;w.E=E;w.N=N;w.S=S;w.texel=texelM/RM*13;w.key=key;w.pend="";w.live=true;w.z=z;
  credits();
}
let winT=0,winBusy=false;
function updateWindows(dt){
  winT-=dt;if(winT>0||winBusy||esriState==="blocked")return;winT=.3;
  const v=sub(cp,EW),dist=len(v),alt=dist-13;
  if(alt>13*3){WIN.forEach(w=>w.live=false);return;}
  // the spot at the centre of the view, else straight below
  const f=dirF(ly,lp),b=dot(v,f),disc=b*b-(dist*dist-169);let pw=v,range=alt;
  if(disc>0&&-b-Math.sqrt(disc)>0){range=-b-Math.sqrt(disc);pw=add(v,mul(f,range));}
  const c=norm(qInv(EQ,pw)),mpp=range/(Math.min(W,H)*1.7)/KM*1000;
  const t0=Math.max(.15,2**Math.floor(Math.log2(mpp*1.1)));
  // coarse levels first, so a new place fills in quickly and sharpens
  for(let k=SATL-1;k>=0;k--){
    const w=WIN[k],texel=t0*4**k;
    if(texel>2400){w.live=false;continue;}
    const far=w.live?Math.acos(clamp(dot(w.C,c),-1,1))*RM>SATN*texel*.22:true;
    if(far||Math.abs(w.texel/13*RM-texel)>1){winBusy=true;buildWin(w,c,texel).finally(()=>{winBusy=false;winT=0;});return;}
  }
}

// the Moon's maria, baked into a small map once (selenographic lat/lon), so the shader does one lookup instead of 37 disc tests
const MOONT=(()=>{const Wm=1024,Hm=512,a=new Uint8Array(Wm*Hm),C=MARIA.map(([la,lo,d])=>[sel(la,lo),d/2/1737.4]);
  for(let j=0;j<Hm;j++){const la=(.5-(j+.5)/Hm)*Math.PI,cl=Math.cos(la),sl=Math.sin(la);
    for(let i=0;i<Wm;i++){const lo=((i+.5)/Wm-.5)*TWO_PI,q=[cl*Math.cos(lo),cl*Math.sin(lo),sl];let m=0;
      for(const [c,r] of C){const d=q[0]*c[0]+q[1]*c[1]+q[2]*c[2];if(d<Math.cos(r*1.12))continue;const th=Math.acos(Math.min(1,d)),x=clamp((r*1.1-th)/(r*.55),0,1);m=Math.max(m,x*x*(3-2*x));}
      a[j*Wm+i]=m*255;}}
  const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);gl.texImage2D(gl.TEXTURE_2D,0,gl.R8,Wm,Hm,0,gl.RED,gl.UNSIGNED_BYTE,a);gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.REPEAT);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  return t;})();

// ---------- the solar system, at true scale, where it really is right now ----------
const AST=window.Astronomy||null;
const AUU=149597870.7*KM,EW=[-36.3,-68,206];           // Earth's centre, relative to the black holes
const SUNW0=norm([.925,.174,-.337]),NW0=norm(sub([0,1,0],mul(SUNW0,SUNW0[1]))),TW0=cross(NW0,SUNW0);
const BODY=[
  {key:"mercury",name:"Mercury",r:2439.7,f:0,p:.142},{key:"venus",name:"Venus",r:6051.8,f:0,p:.689},{key:"earth",name:"Earth",r:6371,f:0,p:.434},
  {key:"moon",name:"Moon",r:1737.4,f:0,p:.12},{key:"mars",name:"Mars",r:3396.2,f:.00589,p:.17},{key:"jupiter",name:"Jupiter",r:71492,f:.06487,p:.538},
  {key:"saturn",name:"Saturn",r:60268,f:.09796,p:.499},{key:"uranus",name:"Uranus",r:25559,f:.02293,p:.488},{key:"neptune",name:"Neptune",r:24764,f:.01708,p:.442}];
const ECLR=AST?AST.Rotation_EQJ_ECL().rot:null;
let eA1=[1,0,0],eA2=[0,1,0];
function e2w(v){const c1=dot(v,eA1),c2=dot(v,eA2),c3=v[2];return[SUNW0[0]*c1+TW0[0]*c2+NW0[0]*c3,SUNW0[1]*c1+TW0[1]*c2+NW0[1]*c3,SUNW0[2]*c1+TW0[2]*c2+NW0[2]*c3];}
const S={sun:[0,0,0],bodies:BODY.map(()=>({pos:[0,0,0],pole:[0,1,0],pm:[1,0,0]})),EQ:[[1,0,0],[0,1,0],[0,0,1]],moonQ:[[1,0,0],[0,1,0],[0,0,1]]};
let EQ=S.EQ,planetsA=null,planetsB=null,planetsT=0;
function axisBasis(north,spinDeg,toW){   // IAU pole + prime-meridian angle -> body-fixed axes in world coordinates
  const n=norm(north),node=norm(cross([0,0,1],n)),W=spinDeg*Math.PI/180;
  const pm=add(mul(node,Math.cos(W)),mul(cross(n,node),Math.sin(W)));
  return{pole:norm(toW(n)),pm:norm(toW(pm))};
}
function planetState(date){
  const T=AST.MakeTime(date),toEcl=v=>mapply(ECLR,[v.x,v.y,v.z]),toW=v=>e2w(mapply(ECLR,v)),out=[];
  for(let i=0;i<BODY.length;i++){
    const b=BODY[i];if(b.key==="earth"||b.key==="moon"){out.push(null);continue;}
    const g=toEcl(AST.GeoVector(b.name,T,true)),ax=AST.RotationAxis(b.name,T),nv=[ax.north.x,ax.north.y,ax.north.z];
    out.push({g,axis:axisBasis(nv,ax.spin,toW)});
  }
  return out;
}
function astro(now){
  const date=new Date(now);
  if(!AST){   // offline fallback: a low-precision Sun and sidereal time, planets hidden
    const d=now/864e5+2440587.5-2451545,g=(357.529+.98560028*d)*Math.PI/180,L=(280.459+.98564736*d+1.915*Math.sin(g)*1+.02*Math.sin(2*g))*Math.PI/180;
    const eps=(23.439-3.6e-7*d)*Math.PI/180,sunE=[Math.cos(L),Math.sin(L),0];eA1=norm(sunE);eA2=cross([0,0,1],eA1);
    S.sun=add(EW,mul(e2w(sunE),AUU));
    const gm=(280.46061837+360.98564736629*d)*Math.PI/180;
    const itrs=v=>{const x=v[0]*Math.cos(gm)-v[1]*Math.sin(gm),y=v[0]*Math.sin(gm)+v[1]*Math.cos(gm),z=v[2];return e2w([x,y*Math.cos(eps)+z*Math.sin(eps),-y*Math.sin(eps)+z*Math.cos(eps)]);};
    S.EQ=[itrs([1,0,0]),itrs([0,0,1]),itrs([0,-1,0])];EQ=S.EQ;S.bodies.forEach((b,i)=>{b.pos=i===2?EW:[1e12,0,0];});return;
  }
  const T=AST.MakeTime(date),toEcl=v=>mapply(ECLR,[v.x,v.y,v.z]);
  const sunE=toEcl(AST.GeoVector("Sun",T,true));eA1=norm([sunE[0],sunE[1],0]);eA2=cross([0,0,1],eA1);
  const toW=v=>e2w(mapply(ECLR,v));
  S.sun=add(EW,mul(e2w(sunE),AUU));
  // Earth's orientation: Greenwich apparent sidereal time, precession and nutation (IAU 2006/2000B)
  const gst=AST.SiderealTime(T)*15*Math.PI/180,Rq=AST.Rotation_EQD_EQJ(T).rot;
  const itrs=v=>toW(mapply(Rq,[v[0]*Math.cos(gst)-v[1]*Math.sin(gst),v[0]*Math.sin(gst)+v[1]*Math.cos(gst),v[2]]));
  S.EQ=[itrs([1,0,0]),itrs([0,0,1]),itrs([0,-1,0])];EQ=S.EQ;
  const m=toEcl(AST.GeoMoon(T));S.bodies[3].pos=add(EW,mul(e2w(m),AUU));
  const ma=AST.RotationAxis("Moon",T),mb=axisBasis([ma.north.x,ma.north.y,ma.north.z],ma.spin,toW);S.bodies[3].pole=mb.pole;S.bodies[3].pm=mb.pm;
  S.bodies[2].pos=EW;
  // the planets move slowly: compute every half second and glide between samples
  if(!planetsA||now>=planetsT+500){planetsA=planetsB&&now<planetsT+1000?planetsB:planetState(new Date(now));planetsB=planetState(new Date(now+500));planetsT=now;}
  const f=clamp((now-planetsT)/500,0,1);
  for(let i=0;i<BODY.length;i++){const a=planetsA[i],b=planetsB[i];if(!a)continue;
    const g=add(mul(a.g,1-f),mul(b.g,f));S.bodies[i].pos=add(EW,mul(e2w(g),AUU));S.bodies[i].pole=a.axis.pole;S.bodies[i].pm=a.axis.pm;}
}
// orbit paths (for the overlay), computed once in ecliptic coordinates
const PERIOD={mercury:87.969,venus:224.701,earth:365.256,mars:686.98,jupiter:4332.59,saturn:10759.22,uranus:30688.5,neptune:60182,moon:27.3217};
let ORB=null;
function buildOrbits(){
  if(!AST)return;ORB=[];const now=Date.now(),N=256;
  for(const b of BODY){const per=PERIOD[b.key];if(!per)continue;const pts=[];
    for(let k=0;k<=N;k++){const T=AST.MakeTime(new Date(now+k/N*per*864e5)),v=b.key==="moon"?AST.GeoMoon(T):AST.HelioVector(b.key==="earth"?"Earth":b.name,T);pts.push(mapply(ECLR,[v.x,v.y,v.z]));}
    ORB.push({key:b.key,geo:b.key==="moon",pts});}
}
setTimeout(buildOrbits,1200);

// ---------- the binary black hole and the AI's machinery ----------
const MH=.32;
let BPH=0,BSEP=3.7,RINGA=0;
const NETA=new Float32Array(NNET*3),NETB=new Float32Array(NNET*3);
const NETDIR=[...Array(8)].map((_,k)=>{const r=Math.sin(k*12.9898+.7)*43758.5453,f=r-Math.floor(r);return(f-.5)*1.1;});
function updateNet(){
  for(let k=0;k<NNET;k++){
    let a,b;
    if(k<8){const th=RINGA+k*Math.PI/4,rad=[Math.cos(th),0,Math.sin(th)],e=NETDIR[k];a=mul(rad,36.3);b=add(a,mul(norm(add(mul(rad,Math.cos(e)),[0,Math.sin(e),0])),6000));}
    else if(k===8){const th=Math.atan2(EW[2],EW[0]);a=[Math.cos(th)*36.3,0,Math.sin(th)*36.3];b=add(EW,mul(norm(sub(a,EW)),13*1.9));}
    else{const th=RINGA+Math.PI/8,rad=[Math.cos(th),0,Math.sin(th)];a=mul(rad,36.3);b=add(a,mul(norm(add(rad,[0,-.8,0])),6000));}
    NETA.set(a,k*3);NETB.set(b,k*3);
  }
}
// the drones: four lights chasing each other round a figure-eight that threads both holes
const HN=256,SPAN=.42;
const LIGHTS=[0,1,2,3].map(k=>({ph:k*Math.PI/2,hue:k*.25,b:9,next:1.5+Math.random()*5,bt:-1,boost:0,hist:new Float32Array(HN*4),hn:0,hi:0,p:[0,0,0]}));
const LP=new Float32Array(NL*NP*4),LB=new Float32Array(NL*4),LC=new Float32Array(NL*4);
function lemPos(ph,k){const A=6.4,s=Math.sin(ph),c=Math.cos(ph),den=1+s*s,qx=A*c/den,qz=A*s*c/den,y=.9*Math.sin(ph*2+k*1.3)+.35*Math.cos(ph);
  const cb=Math.cos(BPH),sb=Math.sin(BPH);return[cb*qx-sb*qz,y,sb*qx+cb*qz];}
function record(L,time){const o=L.hi*4;L.hist[o]=time;L.hist[o+1]=L.p[0];L.hist[o+2]=L.p[1];L.hist[o+3]=L.p[2];L.hi=(L.hi+1)%HN;L.hn=Math.min(HN,L.hn+1);}
function updateLights(dt,now,rate){
  LIGHTS.forEach((L,k)=>{
    L.next-=dt;
    if(L.bt<0&&L.next<=0){L.bt=0;L.bdur=.45+Math.random()*.55;L.peak=2.1+Math.random()*1.2;}
    let speed=1;L.boost=0;
    if(L.bt>=0){L.bt+=dt;const u=L.bt/L.bdur;if(u>=1){L.bt=-1;L.next=2.5+Math.random()*6;}else{L.boost=Math.sin(Math.PI*u);speed=1+(L.peak-1)*L.boost;}}
    L.ph+=dt*.95*speed*rate;L.p=lemPos(L.ph,k);record(L,now);
    const Hs=L.hist,base=k*NP*4;let i=(L.hi-1+HN)%HN,left=L.hn-1;
    for(let j=0;j<NP;j++){
      const tt=now-SPAN*j/(NP-1);let o=(i-1+HN)%HN;
      while(left>0&&Hs[o*4]>tt){i=o;o=(i-1+HN)%HN;left--;}
      const f=left>0?Math.min(1,Math.max(0,(tt-Hs[o*4])/Math.max(1e-6,Hs[i*4]-Hs[o*4]))):1,q=base+j*4;
      for(let c=1;c<4;c++)LP[q+c-1]=left>0?Hs[o*4+c]+(Hs[i*4+c]-Hs[o*4+c])*f:Hs[i*4+c];
    }
    let cx=0,cy=0,cz=0,rad=0;
    for(let j=0;j<NP;j++){const q=base+j*4;cx+=LP[q];cy+=LP[q+1];cz+=LP[q+2];LP[q+3]=j<NP-1?.5*Math.hypot(LP[q+4]-LP[q],LP[q+5]-LP[q+1],LP[q+6]-LP[q+2]):0;}
    cx/=NP;cy/=NP;cz/=NP;
    for(let j=0;j<NP;j++){const q=base+j*4;rad=Math.max(rad,Math.hypot(LP[q]-cx,LP[q+1]-cy,LP[q+2]-cz));}
    LB[k*4]=cx;LB[k*4+1]=cy;LB[k*4+2]=cz;LB[k*4+3]=rad+.35;
    LC[k*4]=L.hue;LC[k*4+1]=L.b*(1+.7*L.boost);LC[k*4+2]=L.boost;
  });
}
for(let i=36;i>0;i--){LIGHTS.forEach((L,k)=>{L.ph+=1/60*.95;L.p=lemPos(L.ph,k);record(L,-i/60);});}

// ---------- palettes, quality ----------
const PAL=[
  {name:"Real",hot:[1,.8,.55],cool:[.55,.12,.02],acc:"#3dff95"},
  {name:"Jarvis green",hot:[.3,1,.6],cool:[.02,.3,.5],acc:"#3dff95"},
  {name:"Event gold",hot:[1,.6,.2],cool:[.6,.12,.04],acc:"#ffb347"},
  {name:"Crimson",hot:[1,.2,.24],cool:[.42,.02,.1],acc:"#ff3f5c"},
  {name:"Nebula",hot:[.62,.45,1],cool:[.12,.14,.62],acc:"#a58bff"},
  {name:"Original B/W",hot:[1,.97,.92],cool:[.34,.33,.33],acc:"#e8e8e8"}];
const hexRGB=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)/255);
const PRESET={auto:{min:.35,max:2,steps:190,ast:64,atm:14,label:"Auto max"},ultra:{min:2,max:2,steps:260,ast:96,atm:20,label:"Ultra 2×"},perf:{min:.6,max:.6,steps:150,ast:40,atm:10,label:"Smooth"},lite:{min:.4,max:.4,steps:110,ast:24,atm:7,label:"Lite"}};
const MODES=["Idle","Listening","Speaking"],SPIN=[1,1.6,2.15],DRIFT=[.05,.09,.07];
const OPT={mode:0,pal:0,q:"auto",cine:false,net:true,orbits:false,stats:false,autohide:true,sun:"live",clouds:true,me:true};
let saved=false;try{const raw=localStorage.getItem("jarvis-bh");if(raw){saved=true;const s=JSON.parse(raw);for(const k in s)if(k in OPT&&k!=="mode")OPT[k]=s[k];}}catch(e){}
function save(){try{const{mode,...rest}=OPT;localStorage.setItem("jarvis-bh",JSON.stringify(rest));}catch(e){}}
if(INTEGRATED&&!saved)OPT.q=SOFTWARE?"lite":"perf";
// the page can ask for the fast GPU but can't pick it: Windows decides per app. Say so, with the fix, when we didn't get it.
if(GPU&&!DISCRETE&&(SOFTWARE||INTEGRATED)){let seen=false;try{seen=sessionStorage.getItem("jarvis-gpu")==="1";}catch(e){}
  if(!seen){$("gpuT").textContent=SOFTWARE?"This is rendering on the CPU, not your RTX":"This is rendering on the integrated GPU, not your RTX";
    $("gpuP").innerHTML=`Your browser handed the page <code>${GPU.replace(/[<>&]/g,"")}</code>. To force the RTX: open <b>Windows Settings → System → Display → Graphics</b>, add your browser (Edge, Chrome, or the Claude app), choose <b>Options → High performance</b>, save, and restart the browser. In Edge or Chrome you can also turn on <code>edge://flags/#force-high-performance-gpu</code> (Chrome: <code>chrome://flags/#force-high-performance-gpu</code>). Quality is set to ${SOFTWARE?"Lite":"Smooth"} until then.`;
    $("gpu").hidden=false;$("gpuX").onclick=()=>{$("gpu").hidden=true;try{sessionStorage.setItem("jarvis-gpu","1");}catch(e){}};}}
let energy=0,voice=0,surge=0,spin=1,spinV=0,diskT=0,last=performance.now(),t=0,frameN=0;
let ss=DISCRETE?1.25:.75,ceiling=2,ceilT=0,adjustT=0,ema=16.7,fpsFrames=0,fpsT=0,fps=0,gpuMs=0,gpuSeen=false;
const queries=[];let dayL=0,dayAuto=false;

// ---------- camera: a frame whose "up" levels itself to the ground near a planet ----------
let cp=[0,0,0],F={X:[1,0,0],U:[0,1,0],Z:[0,0,1]},ly=0,lp=0,tly=0,tlp=0;
const HOME={d:30,el:.36,az:.55};
const homeOffset=()=>[HOME.d*Math.cos(HOME.el)*Math.cos(HOME.az),HOME.d*Math.sin(HOME.el),HOME.d*Math.cos(HOME.el)*Math.sin(HOME.az)];
function dirF(y,p){const cy=Math.cos(y),sy=Math.sin(y),c=Math.cos(p),s=Math.sin(p);return[F.X[0]*c*cy+F.U[0]*s+F.Z[0]*c*sy,F.X[1]*c*cy+F.U[1]*s+F.Z[1]*c*sy,F.X[2]*c*cy+F.U[2]*s+F.Z[2]*c*sy];}
function anglesF(d){return[Math.atan2(dot(d,F.Z),dot(d,F.X)),Math.asin(clamp(dot(d,F.U),-1,1))];}
const clampP=p=>clamp(p,-1.5,1.5),angDiff=(a,b)=>(((b-a)%TWO_PI)+3*Math.PI)%TWO_PI-Math.PI;
function lookAt(target){const d=norm(sub(target,cp));[ly,lp]=anglesF(d);lp=clampP(lp);tly=ly;tlp=lp;}
function rotFrame(k,a){F.X=rotv(F.X,k,a);F.U=rotv(F.U,k,a);F.Z=rotv(F.Z,k,a);}
function retarget(fn){const fw=dirF(ly,lp),tf=dirF(tly,tlp);fn();[ly,lp]=anglesF(fw);[tly,tlp]=anglesF(tf);lp=clampP(lp);tlp=clampP(tlp);}
function levelTo(Ut,maxA){const a=Math.acos(clamp(dot(F.U,Ut),-1,1));if(a<1e-7)return;let k=cross(F.U,Ut);if(len(k)<1e-9)k=F.X;k=norm(k);retarget(()=>rotFrame(k,Math.min(a,maxA)));}
cp=homeOffset();lookAt([0,0,0]);
let lookLatched=false,lookHeld=false,spaceAt=0,flyF=0,orbitV=.05,homeLock=true,travel=null,drag=null,panV=null;
const bodyPos=i=>S.bodies[i].pos,bodyR=i=>BODY[i].r*KM;
function nearestBody(){let best=-1,bd=1e30;for(let i=0;i<BODY.length;i++){const d=(len(sub(cp,bodyPos(i)))-bodyR(i))/bodyR(i);if(d<bd){bd=d;best=i;}}return[best,bd];}
function surfaceDist(){let d=Math.max(len(cp)-4,.05);for(let i=0;i<BODY.length;i++)d=Math.min(d,len(sub(cp,bodyPos(i)))-bodyR(i));d=Math.min(d,len(sub(cp,S.sun))-695700*KM);return Math.max(d,2e-4);}
function keepClear(){
  const r=len(cp);if(r<5.2)cp=mul(cp,5.2/r);
  for(let i=0;i<BODY.length;i++){const c=bodyPos(i),v=sub(cp,c),l=len(v),m=i===2?13+.03*KM:bodyR(i)*(1+4e-4);if(l<m){cp=add(c,mul(v,m/l));flyF=Math.min(flyF,0);}}
  const sv=sub(cp,S.sun),sl=len(sv),sm=695700*KM*1.05;if(sl<sm)cp=add(S.sun,mul(sv,sm/sl));
  if(sl>90*AUU)cp=add(S.sun,mul(sv,90*AUU/sl));
}
const lookOn=()=>lookLatched||lookHeld;
function setLook(on){
  if(on){homeLock=false;travel=null;if(!document.pointerLockElement)try{const p=cv.requestPointerLock();if(p&&p.catch)p.catch(()=>{});}catch(e){}}
  else if(document.pointerLockElement)document.exitPointerLock();
}
addEventListener("mousemove",e=>{if(!lookOn())return;tly+=e.movementX*.0022;tlp=clampP(tlp-e.movementY*.0022);});
function focusBody(){const [i,d]=nearestBody();return d<7?i:-1;}
function orbitHole(da,de){   // swing round the black holes about the world vertical, keeping the same view of them
  const Y=[0,1,0];cp=rotv(cp,Y,-da);rotFrame(Y,-da);
  const r=len(cp),el=Math.asin(clamp(cp[1]/r,-1,1)),el2=clamp(el+de,-1.35,1.35),k=cross(cp,Y);
  if(len(k)>1e-9){const kk=norm(k);cp=rotv(cp,kk,el2-el);rotFrame(kk,el2-el);}
}
function grab(dx,dy,i){   // drag a planet under the cursor, Google-Earth style: turn round its centre
  const c=bodyPos(i),v=sub(cp,c),dist=len(v),up=mul(v,1/dist),R=bodyR(i);
  const fw=dirF(ly,lp),rt=norm(cross(fw,F.U)),uv=cross(rt,fw);
  const rtT=norm(sub(rt,mul(up,dot(rt,up))));let upT=sub(uv,mul(up,dot(uv,up)));upT=len(upT)>1e-6?norm(upT):norm(sub(fw,mul(up,dot(fw,up))));
  const m=add(mul(rtT,-dx),mul(upT,dy)),ml=len(m);if(ml<1e-9)return;
  const b=dot(v,fw),disc=b*b-(dist*dist-R*R),range=disc>0&&-b-Math.sqrt(disc)>0?-b-Math.sqrt(disc):dist-R;
  const ang=Math.min(ml*range/(Math.min(cv.clientWidth,cv.clientHeight)*1.7)/R,.5),k=norm(cross(up,mul(m,1/ml)));
  cp=add(c,rotv(v,k,ang));rotFrame(k,ang);
}
cv.addEventListener("pointerdown",e=>{if(lookOn())return;closeMenus();const fb=focusBody();drag={x:e.clientX,y:e.clientY,t:performance.now(),sx:e.clientX,sy:e.clientY,st:performance.now(),body:fb};orbitV=0;panV=null;homeLock=false;travel=null;cv.setPointerCapture(e.pointerId);});
cv.addEventListener("pointerup",e=>{const d=drag;drag=null;if(d&&Math.hypot(e.clientX-d.sx,e.clientY-d.sy)<6&&performance.now()-d.st<500)tapAt(e.clientX,e.clientY);});cv.addEventListener("pointercancel",()=>drag=null);
cv.addEventListener("pointermove",e=>{
  if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y,now=performance.now(),dts=Math.max(1,now-drag.t)/1000;
  if(drag.body>=0){grab(dx,dy,drag.body);panV={dx:dx/dts,dy:dy/dts,body:drag.body};}
  else{orbitHole(dx*.005,dy*.004);orbitV=orbitV*.5+dx*.005/dts*.5;}
  drag.x=e.clientX;drag.y=e.clientY;drag.t=now;
});
cv.addEventListener("wheel",e=>{e.preventDefault();homeLock=false;travel=null;flyF=clamp(flyF-e.deltaY*(e.deltaMode===1?16:1)*.0045,-3,3);},{passive:false});

// ---------- travel: arcing flights that end facing the target ----------
function startTravel(o){   // o: {anchor(), toLocal(v), toWorld(v), end (local offset), look (local offset), level}
  const s0=o.toLocal(sub(cp,o.anchor()));flyF=0;lookLatched=false;setLook(false);
  const d0=Math.max(len(s0),1e-6),d1=Math.max(len(o.end),1e-6),w=Math.acos(clamp(dot(norm(s0),norm(o.end)),-1,1));
  travel={...o,s0,d0,d1,w,u:0,dur:o.dur||clamp(2.4+.55*Math.abs(Math.log10(d0/d1))+w*1.1,2.4,9),lift:o.lift!=null?o.lift:Math.min(1.6,w*.9)*(d0<d1*40?1:0)};
  homeLock=false;
}
function travelStep(dt){
  const T=travel;T.u=Math.min(1,T.u+dt/T.dur);const u=T.u,e=u<.5?4*u*u*u:1-Math.pow(-2*u+2,3)/2;
  const A=T.anchor(),dir=slerp(norm(T.s0),norm(T.end),e),dist=Math.exp(Math.log(T.d0)*(1-e)+Math.log(T.d1)*e)*(1+T.lift*Math.sin(Math.PI*e));
  cp=add(A,T.toWorld(mul(dir,dist)));
  const L=add(A,T.toWorld(T.look)),k=1-Math.exp(-dt*(2.2+9*u)),[y2,p2]=anglesF(norm(sub(L,cp)));
  tly+=angDiff(tly,y2)*k;tlp+=(clampP(p2)-tlp)*k;ly=tly;lp=tlp;
  if(u>=1){
    if(T.level){const Ut=norm(sub(cp,A));retarget(()=>{});levelTo(Ut,Math.PI);}
    lookAt(L);travel=null;if(T.done)T.done();
  }
}
const IDW={anchor:null,toLocal:v=>v,toWorld:v=>v};
function goHome(){startTravel({...IDW,anchor:()=>[0,0,0],end:homeOffset(),look:[0,0,0],done:()=>{homeLock=true;orbitV=0;}});setTravelUI("hole");}
function goBody(i){
  const c=()=>bodyPos(i),R=bodyR(i),sd=norm(sub(S.sun,bodyPos(i))),away=norm(sub(cp,bodyPos(i)));
  const k=BODY[i].key==="saturn"?5.2:BODY[i].key==="earth"?3.3:3.4,dir=norm(add(add(mul(sd,.72),mul(away,.45)),mul(F.U,.18)));
  startTravel({...IDW,anchor:c,end:mul(dir,R*k),look:[0,0,0],level:false});setTravelUI(BODY[i].key);
}
function goSun(){const away=norm(sub(cp,S.sun)),R=695700*KM;startTravel({...IDW,anchor:()=>S.sun,end:mul(away,R*6),look:[0,0,0]});setTravelUI("sun");}
function goSystem(){   // above the ecliptic, the whole planetary system in view
  const n=e2w([0,0,1]),side=norm(sub(EW,S.sun));
  startTravel({...IDW,anchor:()=>S.sun,end:add(mul(n,78*AUU),mul(side,14*AUU)),look:[0,0,0]});
  if(!OPT.orbits){OPT.orbits=true;syncMenus();}setTravelUI("system");
}
function goEarthPlace(lat,lon,rangeU,tilt,done){   // fly in Earth-fixed coordinates (so the ground doesn't slide), end north-up, tilted
  const up=efDir(lat,lon),E=norm(len(cross([0,1,0],up))>1e-6?cross([0,1,0],up):[0,0,-1]),N=cross(up,E);
  const tp=mul(up,13),camL=add(tp,mul(add(mul(up,Math.cos(tilt)),mul(N,-Math.sin(tilt))),rangeU));
  startTravel({anchor:()=>EW,toLocal:v=>qInv(EQ,v),toWorld:v=>qApply(EQ,v),end:camL,look:tp,level:true,done});setTravelUI("earth");
}
function toggleLook(){lookLatched=!lookLatched;setLook(lookOn());}
function toggleFull(){try{document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen().catch(()=>{});}catch(e){}}
addEventListener("keydown",e=>{
  const typing=e.target&&(e.target.tagName==="INPUT");
  if(e.key==="Escape"){closeMenus();hideResults();if(!typing&&place)closePlace();if(typing)e.target.blur();return;}
  if(typing)return;
  if(e.code==="Space"){e.preventDefault();if(!e.repeat){spaceAt=performance.now();lookHeld=true;setLook(true);}return;}
  const k=e.key.toLowerCase();
  if(k==="/"){e.preventDefault();$("q").focus();return;}
  if(k==="p"){openRoutes("routes");return;}
  if(e.key==="?"){const t=$("tab-keys");if(t)t.click();return;}
  if(k==="h")document.body.classList.toggle("clean");if(k==="f")toggleFull();if(k==="r")goHome();if(k==="e")goBody(2);
  if(k==="o"){OPT.orbits=!OPT.orbits;syncMenus();save();}
  if(k==="1"||k==="2"||k==="3")setMode(+k-1);
  wake();
});
addEventListener("keyup",e=>{if(e.code!=="Space")return;e.preventDefault();lookHeld=false;if(performance.now()-spaceAt<250)lookLatched=!lookLatched;setLook(lookOn());});
document.addEventListener("pointerlockchange",()=>{if(!document.pointerLockElement&&!lookHeld&&lookLatched)lookLatched=false;});
cv.addEventListener("dblclick",()=>{if(!pickable())toggleFull();});
cv.addEventListener("webglcontextlost",e=>{e.preventDefault();ctxLost=true;$("ro1").textContent="The graphics driver reset";$("ro2").textContent="Reloading the scene… If this keeps happening, choose Display → Smooth or Lite.";});cv.addEventListener("webglcontextrestored",()=>location.reload());

// ---------- search: a real geocoder where the page can reach one, Claude where it can't ----------
let sampleFn=null,searchCtl=null,results=[],pin=null;
if(window.claude&&window.claude.use)window.claude.use("sample").then(s=>{sampleFn=s;}).catch(()=>{});
function fetchJSON(url,ms){const ctl=new AbortController(),tm=setTimeout(()=>ctl.abort(),ms);return fetch(url,{signal:ctl.signal,headers:{Accept:"application/json"}}).then(r=>{clearTimeout(tm);if(!r.ok)throw new Error(r.status);return r.json();});}
async function geocode(q){
  try{const j=await fetchJSON(`https://photon.komoot.io/api/?q=${encodeURIComponent(q)}&limit=6&lang=en`,7000);
    const out=(j.features||[]).map(f=>{const p=f.properties||{},[lon,lat]=f.geometry.coordinates,ex=p.extent;
      const detail=[p.street&&(p.housenumber?p.housenumber+" "+p.street:p.street),p.city!==p.name&&p.city,p.state!==p.name&&p.state,p.country!==p.name&&p.country].filter(Boolean).join(", ");
      return{name:p.name||[p.housenumber,p.street].filter(Boolean).join(" ")||detail||q,detail,lat,lon,box:ex?[ex[3],ex[1],ex[0],ex[2]]:null,src:"OpenStreetMap"};});
    if(out.length)return out;}catch(e){}
  try{const j=await fetchJSON(`https://nominatim.openstreetmap.org/search?format=jsonv2&limit=6&q=${encodeURIComponent(q)}`,7000);
    const out=j.map(r=>{const parts=String(r.display_name).split(", ");return{name:r.name||parts[0],detail:parts.slice(1).join(", "),lat:+r.lat,lon:+r.lon,box:r.boundingbox?r.boundingbox.map(Number):null,src:"OpenStreetMap"};});
    if(out.length)return out;}catch(e){}
  if(sampleFn){
    showNote(`<b>Asking Claude where that is…</b> The page can't reach a map search service from here.`);
    const prompt=`You are a precise geocoder. Find the place on Earth that this search refers to and return up to 5 candidates, best match first.
Search: ${JSON.stringify(q)}
Reply with only a JSON array, for example:
[{"name":"Eiffel Tower","detail":"Paris, Île-de-France, France","lat":48.85826,"lon":2.29448,"south":48.8574,"north":48.8591,"west":2.2932,"east":2.2958}]
Rules: WGS84 decimal degrees, as precise as you know them (5 decimals for buildings and landmarks, the centre for cities). The box must tightly cover the whole place (a country's box covers the country). If nothing on Earth matches, reply [].`;
    try{const arr=await sampleFn.json(prompt,{modelTier:"default"});
      if(Array.isArray(arr))return arr.filter(r=>isFinite(+r.lat)&&isFinite(+r.lon)).slice(0,5).map(r=>({name:String(r.name||q),detail:String(r.detail||""),lat:+r.lat,lon:+r.lon,box:[r.south,r.north,r.west,r.east].every(x=>isFinite(+x))?[+r.south,+r.north,+r.west,+r.east]:null,src:"Claude"}));
    }catch(e){
      if(e&&(e.code==="not_granted"||e.code==="sampling_disabled"))throw new Error("Search needs permission to ask Claude. Allow it when asked, or open this page outside the Claude viewer to search OpenStreetMap directly.");
      if(e&&e.code==="rate_limited")throw new Error("Too many searches just now. Try again in a minute.");
      throw new Error("The search didn't come back. Try again, or add a city or country to the search.");
    }
  }
  if(!sampleFn)throw new Error("No search service is reachable from this page. Check your internet connection.");
  return[];
}
function showNote(html){const r=$("results");r.innerHTML=`<div class="note">${html}</div>`;r.hidden=false;}
function hideResults(){$("results").hidden=true;}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]);
function renderResults(list){
  const r=$("results");
  if(!list.length){showNote(`<b>No place found.</b> Try adding a city or country.`);return;}
  r.innerHTML=list.map((x,i)=>`<button type="button" class="res${i===0?" on":""}" data-i="${i}" role="option"><i></i><b>${esc(x.name)}</b><span>${esc(x.detail||"")}${x.detail?" · ":""}${x.lat.toFixed(4)}°, ${x.lon.toFixed(4)}° · ${x.src}</span></button>`).join("");
  r.hidden=false;
  r.querySelectorAll(".res").forEach(b=>b.onclick=()=>{r.querySelectorAll(".res").forEach(x=>x.classList.toggle("on",x===b));flyToPlace(list[+b.dataset.i]);});
}
$("search").addEventListener("submit",async e=>{
  e.preventDefault();const q=$("q").value.trim();if(!q)return;
  showNote("Searching…");
  try{results=await geocode(q);renderResults(results);if(results[0])flyToPlace(results[0]);}
  catch(err){showNote(esc(err.message||"The search failed."));}
});
$("q").addEventListener("focus",()=>{if($("results").innerHTML)$("results").hidden=false;});
function flyToPlace(r){
  const la=r.lat*Math.PI/180,lo=r.lon*Math.PI/180;
  let ext=.5;   // km
  if(r.box&&r.box.every(isFinite)){const[s,n,w,e]=r.box;let dl=e-w;if(dl<0)dl+=360;ext=Math.max((n-s)*110.57,dl*111.32*Math.cos(la),.25);}
  const rangeKm=clamp(ext*2.1,.65,6371*3.2),tilt=rangeKm<60?40*Math.PI/180:rangeKm<2500?25*Math.PI/180:9*Math.PI/180;
  pin={name:r.name,lat:la,lon:lo,detail:r.detail};
  goEarthPlace(la,lo,rangeKm*KM,tilt);
  // it's night there right now? light it so you can see it (switch back under the Earth tab)
  const sunEl=Math.asin(clamp(dot(efDir(la,lo),sunEF(true)),-1,1))*180/Math.PI;
  if(sunEl<-4&&OPT.sun==="live"){OPT.sun="day";dayAuto=true;syncMenus();toast(`<b>It's night in ${esc(r.name)} right now</b> (the Sun is ${Math.abs(sunEl).toFixed(0)}° below the horizon). Lit for viewing. Choose <b>Earth → Real time</b> to see it as it is.`);}
  else if(sunEl>=-4&&dayAuto&&OPT.sun==="day"){OPT.sun="live";dayAuto=false;syncMenus();}
  if(innerWidth<980)hideResults();
}
let toastT=0;function toast(h){const t=$("toast");t.innerHTML=h;t.hidden=false;clearTimeout(toastT);toastT=setTimeout(()=>t.hidden=true,7000);}

// ---------- tap a building: what's there, from OpenStreetMap ----------
const PICK_KM=3,OVERPASS=["https://overpass-api.de/api/interpreter","https://overpass.kumi.systems/api/interpreter"];
let lastView=null,place=null,placeCtl=null,pickHinted=false;
const pickable=()=>len(sub(cp,EW))-13<PICK_KM*KM;
function rayAt(x,y){const V=lastView;if(!V)return null;const m=Math.min(cv.clientWidth,cv.clientHeight)*1.7;
  return norm(add(add(V.fw,mul(V.rt,(x-cv.clientWidth/2)/m)),mul(V.up,-(y-cv.clientHeight/2)/m)));}
function hitEarthT(o,d){const b=dot(o,d),c=dot(o,o)-169,h=b*b-c;if(h<0||b>0||c<0)return -1;return c/(-b+Math.sqrt(h));}
function groundAt(x,y){const d=rayAt(x,y);if(!d)return null;const o=sub(cp,EW),t=hitEarthT(o,d);if(t<0)return null;
  const[la,lo]=efLatLon(norm(qInv(EQ,add(o,mul(d,t)))));return{lat:la*180/Math.PI,lon:lo*180/Math.PI};}
function projGround(la,lo,rt,up,fw){return project(add(EW,qApply(EQ,mul(efDir(la*Math.PI/180,lo*Math.PI/180),13))),rt,up,fw);}
// geometry in metres around a point, good for a few hundred metres
const mx=(lo,lo0,la0)=>(lo-lo0)*111320*Math.cos(la0*Math.PI/180),my=(la,la0)=>(la-la0)*110540;
function ringsOf(e){if(e.geometry)return[e.geometry.map(g=>[g.lat,g.lon])];
  if(e.members)return e.members.filter(m=>m.type==="way"&&m.geometry).map(m=>m.geometry.map(g=>[g.lat,g.lon]));return[];}
function inRings(R,la,lo){let c=false;for(const r of R)for(let i=1;i<r.length;i++){const[y1,x1]=r[i-1],[y2,x2]=r[i];if((y1>la)!==(y2>la)&&lo<(x2-x1)*(la-y1)/(y2-y1)+x1)c=!c;}return c;}
function distM(R,la,lo){let best=1e9;for(const r of R)for(let i=1;i<r.length;i++){
  const ax=mx(r[i-1][1],lo,la),ay=my(r[i-1][0],la),bx=mx(r[i][1],lo,la),by=my(r[i][0],la),vx=bx-ax,vy=by-ay,L=vx*vx+vy*vy||1e-9,u=clamp(-(ax*vx+ay*vy)/L,0,1);
  best=Math.min(best,Math.hypot(ax+vx*u,ay+vy*u));}return best;}
function areaM(r){if(!r||r.length<3)return 1e12;const la0=r[0][0],lo0=r[0][1];let a=0;for(let i=1;i<r.length;i++)a+=mx(r[i-1][1],lo0,la0)*my(r[i][0],la0)-mx(r[i][1],lo0,la0)*my(r[i-1][0],la0);return Math.abs(a)/2;}
const posOf=e=>e.center?[e.center.lat,e.center.lon]:e.lat!=null?[e.lat,e.lon]:null;
const USE=["amenity","shop","tourism","office","leisure","craft","healthcare"];
const rankP=t=>(t.website||t["contact:website"]?4:0)+(t.phone||t["contact:phone"]?2:0)+(USE.some(k=>t[k])?1:0);
function pickPlace(els,la,lo){
  const m=new Map();for(const e of els){const k=e.type+e.id;m.set(k,Object.assign(m.get(k)||{},e));}
  const all=[...m.values()].filter(e=>e.tags);
  const blds=all.filter(e=>e.tags.building&&ringsOf(e).length);
  let B=null,bestA=Infinity;
  for(const e of blds){const R=ringsOf(e);if(inRings(R,la,lo)){const a=areaM(R[0]);if(a<bestA){bestA=a;B=e;}}}
  if(!B){let bd=6;for(const e of blds){const d=distM(ringsOf(e),la,lo);if(d<bd){bd=d;B=e;}}}   // a click just off the roof edge
  const named=all.filter(e=>e.tags.name&&e!==B);let list=[];
  if(B){const R=ringsOf(B);list=named.filter(e=>{const p=posOf(e);return p&&inRings(R,p[0],p[1]);}).sort((a,b)=>rankP(b.tags)-rankP(a.tags));
    if(B.tags.name)list.unshift(B);}
  else{let bd=30,best=null;for(const e of named){if(!USE.some(k=>e.tags[k])&&!e.tags.historic)continue;const p=posOf(e);if(!p)continue;const d=Math.hypot(mx(p[1],lo,la),my(p[0],la));if(d<bd){bd=d;best=e;}}if(best)list=[best];}
  return{B,list};
}
async function overpass(q,signal){let err;
  for(const u of OVERPASS){try{const r=await fetch(u,{method:"POST",body:"data="+encodeURIComponent(q),headers:{"Content-Type":"application/x-www-form-urlencoded"},signal});
    if(!r.ok)throw new Error("HTTP "+r.status);return await r.json();}catch(e){if(signal.aborted)throw e;err=e;}}
  throw err;}
async function reverseAt(la,lo,signal){const r=await fetch(`https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${la.toFixed(7)}&lon=${lo.toFixed(7)}&zoom=18&addressdetails=1`,{signal,headers:{Accept:"application/json"}});
  if(!r.ok)return null;const j=await r.json();return j&&j.address?j:null;}
const hasAddr=t=>!!(t&&(t["addr:housenumber"]||t["addr:street"]));
function addrOf(t){if(!hasAddr(t))return"";const l1=[t["addr:housenumber"],t["addr:street"]].filter(Boolean).join(" ");
  return[l1,t["addr:unit"]&&"Unit "+t["addr:unit"],t["addr:city"],[t["addr:state"],t["addr:postcode"]].filter(Boolean).join(" ")].filter(Boolean).join(", ");}
function revAddr(a){const l1=[a.house_number,a.road].filter(Boolean).join(" ");
  return[l1,a.city||a.town||a.village||a.hamlet||a.suburb,[a.state,a.postcode].filter(Boolean).join(" ")].filter(Boolean).join(", ");}
function kindOf(t){const k=[...USE,"historic","man_made","building"].find(k=>t[k]);if(!k)return"Place";let v=String(t[k]);if(v==="yes")v=k==="building"?"Building":k;
  v=v.replace(/_/g," ");v=v[0].toUpperCase()+v.slice(1);if(t.cuisine)v+=" · "+t.cuisine.split(";")[0].replace(/_/g," ");return v;}
function webOf(t){let u=String(t.website||t["contact:website"]||t.url||t["brand:website"]||"").split(";")[0].trim();if(!u)return"";
  if(!/^https?:\/\//i.test(u))u="https://"+u.replace(/^\/+/,"");try{const x=new URL(u);return /^https?:$/.test(x.protocol)?x.href:"";}catch(e){return"";}}
const phonesOf=t=>[t.phone,t["contact:phone"],t["contact:mobile"]].filter(Boolean).join(";").split(";").map(s=>s.trim()).filter((v,i,a)=>v&&a.indexOf(v)===i);
function tapAt(x,y){if(!pickable())return;const g=groundAt(x,y);if(g)identify(g.lat,g.lon);}
async function identify(la,lo){
  if(placeCtl)placeCtl.abort();const ctl=placeCtl=new AbortController(),tm=setTimeout(()=>ctl.abort(),20000);
  place={lat:la,lon:lo,state:"loading",rings:null,list:[],sel:0};renderPlace();credits();
  const q=`[out:json][timeout:15];
(way(around:45,${la.toFixed(7)},${lo.toFixed(7)})["building"];relation(around:45,${la.toFixed(7)},${lo.toFixed(7)})["building"];)->.b;
.b out tags geom;
nwr(around:70,${la.toFixed(7)},${lo.toFixed(7)})["name"]["highway"!~"."]["boundary"!~"."]["route"!~"."]["place"!~"."]["natural"!~"."]["waterway"!~"."]->.p;
.p out tags center;`;
  try{
    const j=await overpass(q,ctl.signal);if(placeCtl!==ctl)return;
    const{B,list}=pickPlace(j.elements||[],la,lo);
    Object.assign(place,{state:"done",B,list,rings:B?ringsOf(B):null});renderPlace();
    const t=list[0]?list[0].tags:B?B.tags:null;
    if(!hasAddr(t)&&!(B&&hasAddr(B.tags))){const p=list[0]&&posOf(list[0])||[la,lo];
      const r=await reverseAt(p[0],p[1],ctl.signal).catch(()=>null);if(placeCtl===ctl&&r){place.rev=r;renderPlace();}}
  }catch(e){if(placeCtl!==ctl)return;place.state=ctl.signal.aborted?"timeout":"error";renderPlace();}
  finally{clearTimeout(tm);}
}
function closePlace(){if(placeCtl)placeCtl.abort();placeCtl=null;place=null;renderPlace();credits();}
function copyBtn(v){return`<button type="button" class="pc" data-copy="${esc(v)}">Copy</button>`;}
function renderPlace(){
  const el=$("place"),b=$("placeBody");if(!place){el.hidden=true;return;}el.hidden=false;
  const ll=`${Math.abs(place.lat).toFixed(6)}°${place.lat>=0?"N":"S"} ${Math.abs(place.lon).toFixed(6)}°${place.lon>=0?"E":"W"}`,here=`${place.lat.toFixed(6)},${place.lon.toFixed(6)}`;
  const gHere=`https://www.google.com/maps/search/?api=1&query=${here}`;
  if(place.state==="loading"){b.innerHTML=`<div class="pk"><i class="spin"></i>Looking up this spot</div><h2 class="pn">Checking OpenStreetMap…</h2><p class="pm">${ll}</p>`;return;}
  if(place.state!=="done"){const blocked=esriState==="blocked";
    b.innerHTML=`<div class="pk">Details unavailable</div><h2 class="pn">${blocked?"Building lookup can't run in the Claude viewer":place.state==="timeout"?"OpenStreetMap took too long":"Couldn't reach OpenStreetMap"}</h2>
      <p class="pp">${blocked?"The viewer blocks outside map services. Open the downloaded copy of this page in Edge to click buildings. Google Maps can show you this exact spot meanwhile.":"Check your internet connection and try again, or open this spot on Google Maps."}</p>
      <div class="pa"><a class="pb pri" href="${gHere}" target="_blank" rel="noopener">This spot on Google Maps ↗</a>${blocked?"":`<button class="pb" type="button" data-retry>Try again</button>`}</div><p class="pm">${ll}</p>`;return;}
  const{B,list}=place,cur=list[place.sel]||null,t=cur?cur.tags:B?B.tags:{};
  const name=t.name||(B?"Unnamed building":"No listed place here");
  let addr=addrOf(t)||(B?addrOf(B.tags):""),approx=false;
  if(!addr&&place.rev){addr=revAddr(place.rev.address);approx=true;}
  const phones=phonesOf(t),web=webOf(t),hours=t.opening_hours||"",brand=t.brand&&t.brand!==t.name?t.brand:t.operator&&t.operator!==t.name?t.operator:"";
  const qName=t.name?encodeURIComponent(t.name+(addr?", "+addr:"")):here;
  const src=cur||B,osm=src?`https://www.openstreetmap.org/${src.type}/${src.id}`:`https://www.openstreetmap.org/#map=19/${here.replace(",","/")}`;
  const rows=[];
  if(addr)rows.push(`<div class="pr"><dt>${approx?"Nearest address":"Address"}</dt><dd><span>${esc(addr)}</span>${copyBtn(addr)}</dd></div>`);
  phones.forEach((p,i)=>rows.push(`<div class="pr"><dt>${i?"Phone "+(i+1):"Phone"}</dt><dd><a href="tel:${esc(p.replace(/[^\d+]/g,""))}">${esc(p)}</a>${copyBtn(p)}</dd></div>`));
  if(hours)rows.push(`<div class="pr"><dt>Hours</dt><dd><span>${esc(hours.replace(/;\s*/g,"; "))}</span></dd></div>`);
  if(!addr&&!phones.length&&t.name)rows.push(`<div class="pr"><dt>Listing</dt><dd><span class="dimv">No address or phone number is listed for this place.</span></dd></div>`);
  if(!addr&&!place.rev&&!t.name)rows.push(`<div class="pr"><dt>Address</dt><dd><span class="dimv">Looking up the nearest address…</span></dd></div>`);
  const others=list.map((e,i)=>i===place.sel?"":`<button type="button" class="po" data-sel="${i}">${esc(e.tags.name)}<em>${esc(kindOf(e.tags))}</em></button>`).join("");
  b.innerHTML=`<div class="pk">${esc(src?kindOf(t):"Open ground")}${brand?" · "+esc(brand):""}</div><h2 class="pn">${esc(name)}</h2>
    ${rows.length?`<dl class="pd">${rows.join("")}</dl>`:""}
    <div class="pa">${web?`<a class="pb pri" href="${esc(web)}" target="_blank" rel="noopener">Visit website ↗</a>`:""}
      <a class="pb${web?"":" pri"}" href="https://www.google.com/maps/search/?api=1&query=${qName}" target="_blank" rel="noopener">Google Maps ↗</a>
      ${!web&&t.name?`<a class="pb" href="https://www.google.com/search?q=${qName}" target="_blank" rel="noopener">Search the web ↗</a>`:""}
      <a class="pb" href="${osm}" target="_blank" rel="noopener">OpenStreetMap ↗</a></div>
    <div class="pa pa2"><button type="button" class="pb" data-save>${LM.some(l=>Math.abs(l.lat-place.lat)<5e-5&&Math.abs(l.lon-place.lon)<5e-5)?"Saved as landmark ✓":"Save as landmark"}</button><button type="button" class="pb" data-routehere>Route here</button></div>
    ${others?`<div class="pl2">${list.length>1&&B?"Also in this building":"Also here"}</div><div class="pos">${others}</div>`:""}
    <p class="pm">${ll} · Listing from OpenStreetMap contributors${approx?" · address from Nominatim":""}. Details can be out of date.</p>`;
}
$("placeX").addEventListener("click",closePlace);
$("placeBody").addEventListener("click",e=>{
  const c=e.target.closest("[data-copy]");if(c){const v=c.dataset.copy;const ok=()=>{c.textContent="Copied";setTimeout(()=>c.textContent="Copy",1400);};
    try{navigator.clipboard.writeText(v).then(ok,()=>{const s=c.previousElementSibling;if(s){const r=document.createRange();r.selectNodeContents(s);const sel=getSelection();sel.removeAllRanges();sel.addRange(r);}});}catch(err){}return;}
  const s=e.target.closest("[data-sel]");if(s){place.sel=+s.dataset.sel;renderPlace();return;}
  if(e.target.closest("[data-retry]"))identify(place.lat,place.lon);
  if(e.target.closest("[data-save]")){if(!LM.some(l=>Math.abs(l.lat-place.lat)<5e-5&&Math.abs(l.lon-place.lon)<5e-5))addLandmark(placeName()||"Saved spot",place.lat,place.lon);renderPlace();return;}
  if(e.target.closest("[data-routehere]")){RT.to="spot";RT.tab="routes";openRoutes("routes");plotRoute();}
});
function drawPlace(rt,up,fw){
  const svg=$("pickSvg");
  if(!place||len(sub(cp,EW))-13>60*KM){svg.hidden=true;return;}
  svg.hidden=false;let d="";
  for(const r of place.rings||[]){let pen=false;for(const[la,lo]of r){const q=projGround(la,lo,rt,up,fw);if(!q){pen=false;continue;}d+=(pen?"L":"M")+q[0].toFixed(1)+" "+q[1].toFixed(1);pen=true;}}
  $("pickPath").setAttribute("d",d);
  const m=projGround(place.lat,place.lon,rt,up,fw),dot_=$("pickDot");
  if(m){dot_.setAttribute("cx",m[0].toFixed(1));dot_.setAttribute("cy",m[1].toFixed(1));dot_.removeAttribute("visibility");}else dot_.setAttribute("visibility","hidden");
}
// is any part of the core (holes, disk, ring, jets) on screen and not behind the Earth?
const CORE_PTS=[[0,0,0],[0,60,0],[0,-60,0],[0,160,0],[0,-160,0],[0,250,0],[0,-250,0],[40,0,0],[-40,0,0],[0,0,40],[0,0,-40]];
function coreVisible(rt,up,fw){
  if(len(cp)<400)return true;const W_=cv.clientWidth,H_=cv.clientHeight,o=sub(cp,EW);
  for(const p of CORE_PTS){const q=project(p,rt,up,fw);if(!q||q[0]<-W_*.15||q[0]>W_*1.15||q[1]<-H_*.15||q[1]>H_*1.15)continue;
    const v=sub(p,cp),D=len(v),t=hitEarthT(o,mul(v,1/D));if(t<0||t>D)return true;}
  return false;
}
// the Earth covers the whole frame: nothing beyond it can show
function earthFills(){if(len(sub(cp,EW))>13*3)return false;const o=sub(cp,EW),W_=cv.clientWidth,H_=cv.clientHeight;
  return[[0,0],[W_,0],[0,H_],[W_,H_],[W_/2,0]].every(([x,y])=>{const d=rayAt(x,y);return d&&hitEarthT(o,d)>=0;});}

// ---------- menus ----------
const KEYLIST=[["Keyboard",[["/","Search Earth"],["P","Routes and landmarks"],["E","Fly to Earth"],["R","Back to the black hole"],["O","Orbits and labels"],["1 2 3","Jarvis mode: idle, listening, speaking"],["H","Hide everything"],["F","Full screen"],["Space","Look around (tap to keep it on)"],["Esc","Close menus and cards"],["?","This list"]]],
  ["Mouse",[["Drag","Orbit the core, or spin a planet under you"],["Scroll","Fly forward and back"],["Click","Below 3 km: a building's name, address, phone and website"],["Double-click","Full screen (when not zoomed in on Earth)"]]]];
const MENUS=[
  {id:"mode",label:"Mode",val:()=>MODES[OPT.mode],groups:[{title:"Jarvis state",radio:"mode",items:MODES.map((m,i)=>[i,m,["1","2","3"][i]])}]},
  {id:"travel",label:"Travel",val:()=>"",groups:[{title:"Go to",act:"travel",items:[["hole","Binary core","R"],["sun","Sun"],["mercury","Mercury"],["venus","Venus"],["earth","Earth","E"],["moon","Moon"],["mars","Mars"],["jupiter","Jupiter"],["saturn","Saturn"],["uranus","Uranus"],["neptune","Neptune"],["system","Whole solar system"]]}]},
  {id:"style",label:"Style",val:()=>"",groups:[{title:"Palette",radio:"pal",items:PAL.map((p,i)=>[i,p.name])},{title:"Overlays",check:[["net","AI network across the sky"],["cine","Cinema bars (2.39:1)"]]}]},
  {id:"earth",label:"Earth",val:()=>OPT.sun==="live"?"Live":"Day",groups:[{title:"Lighting",radio:"sun",items:[["live","Real time · Sun and Moon"],["day","Always daylight"]]},{title:"Right now",info:"here"},{title:"Show",check:[["clouds","Clouds"],["me","My location"]]},{title:"Imagery",info:"imagery"}]},
  {id:"display",label:"Display",val:()=>"",groups:[{title:"Quality",radio:"q",items:Object.entries(PRESET).map(([k,v])=>[k,v.label])},{title:"Show",check:[["orbits","Orbits and labels","O"],["stats","Performance stats"],["autohide","Hide controls when idle"]]},{title:"Graphics card",info:"gpu"},{title:"Controls",keys:true}]},
  {id:"keys",label:"Keys",val:()=>"",groups:KEYLIST.map(([t,l])=>({title:t,keylist:l}))}];
let travelName="Binary core",openMenu=null;
const chev=`<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>`;
function buildMenus(){
  const nav=$("tabs");nav.innerHTML="";
  for(const m of MENUS){
    const dd=document.createElement("div");dd.className="dd";
    dd.innerHTML=`<button class="tab" type="button" aria-haspopup="true" aria-expanded="false" id="tab-${m.id}">${m.label} <b class="val" id="val-${m.id}"></b>${chev}</button><div class="menu" role="menu" hidden id="menu-${m.id}"></div>`;
    nav.appendChild(dd);
    const menu=dd.querySelector(".menu");
    m.groups.forEach((g,gi)=>{
      if(gi)menu.insertAdjacentHTML("beforeend",`<div class="sep"></div>`);
      menu.insertAdjacentHTML("beforeend",`<div class="lbl">${g.title}</div>`);
      if(g.radio)g.items.forEach(([v,label,key])=>{const sw=g.radio==="pal"?`<span class="sw" style="background:linear-gradient(135deg,rgb(${PAL[v].hot.map(x=>x*255|0)}),rgb(${PAL[v].cool.map(x=>x*255|0)}) 60%,${PAL[v].acc})"></span>`:`<span class="ck"></span>`;
        menu.insertAdjacentHTML("beforeend",`<button class="opt" type="button" role="menuitemradio" data-k="${g.radio}" data-v="${v}">${sw}${label}${key?`<span class="meta"><kbd>${key}</kbd></span>`:""}</button>`);});
      if(g.act)g.items.forEach(([v,label,key])=>menu.insertAdjacentHTML("beforeend",`<button class="opt" type="button" role="menuitem" data-go="${v}"><span class="ck"></span>${label}<span class="meta" data-dist="${v}">${key?`<kbd>${key}</kbd>`:""}</span></button>`));
      if(g.check)g.check.forEach(([k,label,key])=>menu.insertAdjacentHTML("beforeend",`<button class="opt" type="button" role="menuitemcheckbox" data-c="${k}"><span class="tg"></span>${label}${key?`<span class="meta"><kbd>${key}</kbd></span>`:""}</button>`));
      if(g.info==="here")menu.insertAdjacentHTML("beforeend",`<div class="info" id="hereInfo"></div>`);
      if(g.info==="imagery")menu.insertAdjacentHTML("beforeend",`<div class="info" id="imgInfo"></div>`);
      if(g.info==="gpu")menu.insertAdjacentHTML("beforeend",`<div class="info"><b>${esc(GPU||"Unknown")}</b><br>${DISCRETE?"Rendering on the dedicated GPU.":SOFTWARE?"Software rendering on the CPU: see Windows Settings → Display → Graphics to force the RTX.":"Integrated GPU: set your browser to High performance in Windows Settings → Display → Graphics to use the RTX."}</div>`);
      if(g.keylist)menu.insertAdjacentHTML("beforeend",`<div class="keys">${g.keylist.map(([k,d])=>`<span class="kk">${k.split(" ").map(x=>`<kbd>${x}</kbd>`).join("")}</span><span>${d}</span>`).join("")}</div>`);
      if(g.keys)menu.insertAdjacentHTML("beforeend",`<div class="keys"><kbd>Drag</kbd><span>Orbit the core, or spin a planet under you</span><kbd>Scroll</kbd><span>Fly forward and back</span><kbd>Space</kbd><span>Look around (tap to keep it on)</span><kbd>/</kbd><span>Search Earth</span><kbd>P</kbd><span>Routes and landmarks</span><kbd>H</kbd><span>Hide everything</span><kbd>F</kbd><span>Full screen</span></div>`);
    });
    const tab=dd.querySelector(".tab");
    tab.onclick=e=>{e.stopPropagation();const open=menu.hidden;closeMenus();if(open){menu.hidden=false;tab.setAttribute("aria-expanded","true");openMenu=m.id;if(m.id==="travel")travelDistances();if(m.id==="earth"){imageryInfo();hereInfo();}}};
    menu.addEventListener("click",e=>{
      const b=e.target.closest(".opt");if(!b)return;e.stopPropagation();
      if(b.dataset.k){const k=b.dataset.k,v=b.dataset.v;
        if(k==="mode")setMode(+v);else if(k==="pal"){OPT.pal=+v;}else if(k==="q"){OPT.q=v;ceiling=PRESET[v].max;ss=clamp(ss,PRESET[v].min,PRESET[v].max);}else if(k==="sun"){OPT.sun=v;dayAuto=false;}
        save();syncMenus();}
      if(b.dataset.c){OPT[b.dataset.c]=!OPT[b.dataset.c];save();syncMenus();}
      if(b.dataset.go){closeMenus();const g=b.dataset.go;if(g==="hole")goHome();else if(g==="sun")goSun();else if(g==="system")goSystem();else goBody(BODY.findIndex(x=>x.key===g));}
    });
  }
  const pb=document.createElement("button");pb.className="tab places";pb.type="button";pb.id="tab-places";pb.setAttribute("aria-expanded","false");
  pb.innerHTML=`Places <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.4"/></svg>`;
  pb.onclick=e=>{e.stopPropagation();closeMenus();toggleRoutes();};nav.prepend(pb);
  const fs=document.createElement("button");fs.className="tab icon";fs.type="button";fs.title="Full screen (F)";fs.setAttribute("aria-label","Full screen");
  fs.innerHTML=`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>`;fs.onclick=toggleFull;nav.appendChild(fs);
  syncMenus();
}
function closeMenus(){document.querySelectorAll(".menu").forEach(m=>m.hidden=true);document.querySelectorAll(".tab[aria-expanded]").forEach(t=>t.setAttribute("aria-expanded","false"));openMenu=null;}
document.addEventListener("click",e=>{if(!e.target.closest(".dd"))closeMenus();if(!e.target.closest(".search"))hideResults();});
function syncMenus(){
  document.querySelectorAll(".opt[data-k]").forEach(b=>b.setAttribute("aria-checked",String(String(OPT[b.dataset.k])===b.dataset.v)));
  document.querySelectorAll(".opt[data-c]").forEach(b=>b.setAttribute("aria-checked",String(!!OPT[b.dataset.c])));
  document.querySelectorAll(".opt[data-go]").forEach(b=>b.setAttribute("aria-checked",String(b.dataset.go===travelKey)));
  for(const m of MENUS){const el=$("val-"+m.id);if(el)el.textContent=m.val();}
  document.documentElement.style.setProperty("--accent",PAL[OPT.pal].acc);
  $("stats").hidden=!OPT.stats;
  $("stMode").textContent=MODES[OPT.mode].toUpperCase();
  credits();
}
let travelKey="hole";
function setTravelUI(k){travelKey=k;const it=MENUS[1].groups[0].items.find(x=>x[0]===k);travelName=it?it[1]:"Free flight";syncMenus();}
function setMode(m){if(m!==OPT.mode)surge=1;OPT.mode=m;syncMenus();}
const fmtDist=km=>km<1?`${(km*1000).toFixed(0)} m`:km<1e5?`${km.toLocaleString(undefined,{maximumFractionDigits:km<100?1:0})} km`:km<1.5e7?`${(km/1e6).toFixed(2)}M km`:`${(km/149597870.7).toFixed(2)} AU`;
function travelDistances(){
  document.querySelectorAll("[data-dist]").forEach(el=>{const k=el.dataset.dist;let p=null;
    if(k==="hole")p=[0,0,0];else if(k==="sun")p=S.sun;else{const i=BODY.findIndex(b=>b.key===k);if(i>=0)p=bodyPos(i);}
    if(p){const km=len(sub(p,cp))/KM;el.textContent=fmtDist(km);}});
}
function imageryInfo(){const el=$("imgInfo");if(!el)return;
  el.innerHTML=esriState==="ok"?`<b>High-resolution satellite imagery is streaming</b> (Esri World Imagery, up to about 0.3 m per pixel). NASA Blue Marble fills in the rest.`
    :esriState==="blocked"?`<b>Street-level imagery can't load inside the Claude viewer</b>, which blocks outside map servers. You're seeing NASA Blue Marble at 7 km per pixel. Open the downloaded copy of this page in your browser for sharp imagery down to about 0.3 m.`
    :`Checking for high-resolution imagery… NASA Blue Marble (7 km per pixel) is showing meanwhile.`;
  el.innerHTML+=`<br><br>Clouds: ${cloudLive?`<b>live</b>, from EUMETSAT satellites, updated every 3 hours (last loaded ${new Date(cloudStamp).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})})`:"a real satellite cloud map (not today's weather; live clouds load when this page can reach the internet)"}.`;
}
function credits(){
  const parts=["Earth: NASA Blue Marble &amp; Black Marble"];
  if(esriState==="ok"&&WIN.some(w=>w.live))parts.push("Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community");
  if(cloudLive)parts.push("Clouds: contains modified EUMETSAT data");
  if(results.length&&results[0].src==="OpenStreetMap"||place)parts.push("Search &amp; places: © OpenStreetMap contributors");
  if(AST)parts.push("Positions: Astronomy Engine");
  $("credits").innerHTML=parts.join("<br>");
}
buildMenus();

// ---------- auto-hide the controls when you stop moving ----------
let idleT=0;function wake(){idleT=0;document.body.classList.remove("idle");}
["mousemove","pointerdown","wheel","touchstart"].forEach(ev=>addEventListener(ev,wake,{passive:true}));


// ---------- your location: Ephrata, PA, lit exactly as it is right now ----------
const ME={name:"Ephrata, PA",lat:40.1798*Math.PI/180,lon:-76.1788*Math.PI/180};
const meFmt=new Intl.DateTimeFormat("en-US",{timeZone:"America/New_York",hour:"numeric",minute:"2-digit"});
function skyAt(la,lo){   // elevations in degrees, topocentric, from the real Sun and Moon positions
  const up=efDir(la,lo),sun=Math.asin(clamp(dot(up,sunEF(true)),-1,1))*180/Math.PI;
  const mP=S.bodies[3].pos,mv=norm(sub(qInv(EQ,sub(mP,EW)),mul(up,13))),moon=Math.asin(clamp(dot(up,mv),-1,1))*180/Math.PI;
  const lit=(1+dot(norm(sub(S.sun,mP)),norm(sub(EW,mP))))/2;return{sun,moon,lit};}
const phaseOf=el=>el>0?"Daytime":el>-6?"Civil twilight":el>-12?"Nautical twilight":el>-18?"Astronomical twilight":"Night";
function hereInfo(){const el=$("hereInfo");if(!el)return;const k=skyAt(ME.lat,ME.lon);
  el.innerHTML=`<b>${phaseOf(k.sun)} in ${ME.name}</b> · ${meFmt.format(new Date())}<br>Sun ${k.sun>=0?"+":""}${k.sun.toFixed(1)}° · Moon ${k.moon>=0?"+":""}${k.moon.toFixed(1)}°, ${Math.round(k.lit*100)}% lit`+
    (OPT.sun==="live"?`<br>The globe shows this lighting: sunlight, twilight and moonlight where they fall at this moment.`:`<br>Daylight is forced on. Choose <b>Real time</b> to see the planet as it is now.`);}

// ---------- live weather in Ephrata, the same pill as the Punch In dashboard ----------
const WX_ICONS={
  sun:'<svg class="wx-icon" viewBox="0 0 24 24" fill="none" stroke="#e8b15c" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.2" fill="#e8b15c" stroke="none"/><line x1="12" y1="2.4" x2="12" y2="4.6"/><line x1="12" y1="19.4" x2="12" y2="21.6"/><line x1="2.4" y1="12" x2="4.6" y2="12"/><line x1="19.4" y1="12" x2="21.6" y2="12"/><line x1="5.2" y1="5.2" x2="6.8" y2="6.8"/><line x1="17.2" y1="17.2" x2="18.8" y2="18.8"/><line x1="18.8" y1="5.2" x2="17.2" y2="6.8"/><line x1="6.8" y1="17.2" x2="5.2" y2="18.8"/></svg>',
  moon:'<svg class="wx-icon" viewBox="0 0 24 24" fill="#d8d2c2" stroke="none"><path d="M15.5 3.2a8.6 8.6 0 1 0 5.3 13.9A7.4 7.4 0 0 1 15.5 3.2z"/></svg>',
  cloud:'<svg class="wx-icon" viewBox="0 0 24 24" fill="none" stroke="#a0959a" stroke-width="2" stroke-linejoin="round"><path d="M6.5 18h11a3.9 3.9 0 0 0 .2-7.8 5.6 5.6 0 0 0-10.7-1.3A3.9 3.9 0 0 0 6.5 18z"/></svg>',
  part:'<svg class="wx-icon" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="8" r="3.2" fill="#e8b15c" stroke="none"/><path d="M9 18h9a3.4 3.4 0 0 0 .2-6.8 4.9 4.9 0 0 0-9.3-1.1A3.4 3.4 0 0 0 9 18z" stroke="#a0959a"/></svg>',
  rain:'<svg class="wx-icon" viewBox="0 0 24 24" fill="none" stroke="#b3a4dd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6.5 15h11a3.9 3.9 0 0 0 .2-7.8 5.6 5.6 0 0 0-10.7-1.3A3.9 3.9 0 0 0 6.5 15z"/><line x1="9" y1="18" x2="8" y2="21"/><line x1="13" y1="18" x2="12" y2="21"/><line x1="17" y1="18" x2="16" y2="21"/></svg>',
  storm:'<svg class="wx-icon" viewBox="0 0 24 24" fill="none" stroke="#b3a4dd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6.5 14h11a3.9 3.9 0 0 0 .2-7.8 5.6 5.6 0 0 0-10.7-1.3A3.9 3.9 0 0 0 6.5 14z"/><polygon points="13,16 9.5,21.5 12,21.5 10.5,24.5 15,18.5 12.5,18.5" fill="#ff3f5c" stroke="none"/></svg>',
  fog:'<svg class="wx-icon" viewBox="0 0 24 24" fill="none" stroke="#a0959a" stroke-width="2" stroke-linecap="round"><line x1="3.5" y1="9" x2="20.5" y2="9"/><line x1="5.5" y1="13" x2="18.5" y2="13"/><line x1="3.5" y1="17" x2="20.5" y2="17"/></svg>',
  snow:'<svg class="wx-icon" viewBox="0 0 24 24" fill="none" stroke="#a0959a" stroke-width="2" stroke-linecap="round"><path d="M6.5 14h11a3.9 3.9 0 0 0 .2-7.8 5.6 5.6 0 0 0-10.7-1.3A3.9 3.9 0 0 0 6.5 14z"/><line x1="9" y1="18.5" x2="9" y2="18.6"/><line x1="13" y1="20.5" x2="13" y2="20.6"/><line x1="16.5" y1="18.5" x2="16.5" y2="18.6"/></svg>'};
function wxIconFor(text,night){const t=String(text||"").toLowerCase();
  if(/thunder|storm/.test(t))return WX_ICONS.storm;if(/snow|sleet|flurr/.test(t))return WX_ICONS.snow;if(/rain|shower|drizzle/.test(t))return WX_ICONS.rain;
  if(/fog|haze|mist/.test(t))return WX_ICONS.fog;if(/partly|mostly sunny|mostly clear|few clouds/.test(t))return WX_ICONS.part;if(/cloud|overcast/.test(t))return WX_ICONS.cloud;
  if(/fair|clear|sunny/.test(t))return night?WX_ICONS.moon:WX_ICONS.sun;return WX_ICONS.cloud;}
const WMO={0:"Clear",1:"Mostly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",48:"Freezing fog",51:"Light drizzle",53:"Drizzle",55:"Heavy drizzle",61:"Light rain",63:"Rain",65:"Heavy rain",
  66:"Freezing rain",67:"Freezing rain",71:"Light snow",73:"Snow",75:"Heavy snow",77:"Snow grains",80:"Rain showers",81:"Rain showers",82:"Heavy showers",85:"Snow showers",86:"Snow showers",
  95:"Thunderstorms",96:"Thunderstorms with hail",99:"Thunderstorms with hail"};
let WX=null;const TZNY="America/New_York";
function renderWeather(wx){
  const host=$("wx");if(!wx){return;}WX=wx;host.hidden=false;
  const night=skyAt(ME.lat,ME.lon).sun<-.8;
  $("wxIcon").innerHTML=wxIconFor(wx.condition,night);$("wxTemp").textContent=Math.round(wx.tempF)+"°";$("wxPlace").textContent=(wx.place||"").split(",")[0];
  const obs=wx.observedAt?new Date(wx.observedAt.length<=16?wx.observedAt+":00-04:00":wx.observedAt).toLocaleTimeString("en-US",{timeZone:TZNY,hour:"numeric",minute:"2-digit"}):"";
  const cells=[["Now",Math.round(wx.tempF)+"°F"],["Humidity",wx.humidity!=null?wx.humidity+"%":"—"],["Wind",wx.wind||"—"],["Sky",wx.condition||"—"]];
  let h=`<div class="wx-pop-head"><span class="wx-pop-place">${esc(wx.place||"")}</span><span class="wx-pop-obs">${obs?"obs "+obs:""}</span></div><div class="wx-grid">`+
    cells.map(c=>`<div class="wx-cell"><span>${c[0]}</span><span>${esc(c[1])}</span></div>`).join("")+`</div><div class="wx-periods">`+
    (wx.periods||[]).map(p=>`<div><div class="wx-period-name">${esc(p.name)}</div><div class="wx-period-text"><span class="wx-period-temp">${esc(p.temp||"")}</span>${p.temp?" · ":""}${esc(p.text||"")}</div></div>`).join("")+`</div>`;
  if((wx.hourly||[]).length)h+=`<div class="wx-hourly"><div class="wx-hourly-title">Next 6 hours</div><div class="wx-hourly-row">`+wx.hourly.slice(0,6).map(x=>`<div class="wx-hour"><div class="wx-hour-time">${esc(x.time)}</div><div class="wx-hour-icon">${wxIconFor(x.condition,x.night)}</div><div class="wx-hour-temp">${Math.round(x.tempF)}°</div><div class="wx-hour-detail">${x.precipChance!=null?x.precipChance+"% rain":esc(x.condition)}</div></div>`).join("")+`</div></div>`;
  if(wx.station)h+=`<div class="wx-src">${esc(wx.station)}</div>`;
  $("wxPop").innerHTML=h;
}
function fetchLiveWeather(){
  const url="https://api.open-meteo.com/v1/forecast?latitude=40.1801&longitude=-76.1785&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,is_day"+
    "&hourly=temperature_2m,precipitation_probability,weather_code,is_day&daily=weather_code,temperature_2m_max,temperature_2m_min&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=America%2FNew_York&forecast_days=3";
  return fetchJSON(url,12000).then(j=>{if(!j||!j.current)return false;const c=j.current,d=j.daily||{},hh=j.hourly||{};
    const periods=[];for(let i=0;i<Math.min(3,(d.time||[]).length);i++){const dt=new Date(d.time[i]+"T12:00:00");
      periods.push({name:i===0?"Today":dt.toLocaleDateString("en-US",{weekday:"long"}),temp:`High ${Math.round(d.temperature_2m_max[i])}° / Low ${Math.round(d.temperature_2m_min[i])}°`,text:WMO[d.weather_code[i]]||""});}
    const live={place:"Ephrata, PA",station:"Live · Open-Meteo · refreshes every 30 minutes",tempF:c.temperature_2m,condition:WMO[c.weather_code]||"",humidity:c.relative_humidity_2m,wind:Math.round(c.wind_speed_10m)+" mph",observedAt:c.time,periods,hourly:[]};
    if(hh.time){let n=hh.time.indexOf(c.time.slice(0,13)+":00");if(n<0)n=Math.max(0,hh.time.findIndex(t=>t>c.time));
      for(let i=n;i<Math.min(n+6,hh.time.length);i++)live.hourly.push({time:new Date(hh.time[i]+":00-04:00").toLocaleTimeString("en-US",{timeZone:TZNY,hour:"numeric"}),tempF:hh.temperature_2m[i],precipChance:hh.precipitation_probability[i],condition:WMO[hh.weather_code[i]]||"",night:hh.is_day&&hh.is_day[i]===0});}
    renderWeather(live);return true;}).catch(()=>false);
}
// the Punch In dashboard's last synced reading, where live weather can't be reached (inside the Claude viewer)
function storedWeather(){try{if(!(window.claude&&window.claude.use))return;window.claude.use("db").then(db=>{if(!db)return;
  db.doc("dashboard/data").onSnapshot(s=>{const d=s&&s.exists?s.data():null;if(d&&d.weather&&(!WX||/stored/.test(WX.station||"")))renderWeather({...d.weather,place:d.weather.place||"Ephrata, PA",station:(d.weather.station||"Stored reading")+" · stored reading from Punch In"});},()=>{});}).catch(()=>{});}catch(e){}}
fetchLiveWeather().then(ok=>{if(!ok)storedWeather();});setInterval(fetchLiveWeather,18e5);
$("wx").addEventListener("click",e=>{if(e.target.closest(".wx-pop"))return;$("wx").classList.toggle("open");});
document.addEventListener("click",e=>{if(!e.target.closest("#wx"))$("wx").classList.remove("open");});

// ---------- landmarks and routes: how long by foot, skateboard, bike and car ----------
const LMKEY="jarvis-landmarks";let LM=[];
try{LM=(JSON.parse(localStorage.getItem(LMKEY)||"[]")||[]).filter(x=>x&&isFinite(x.lat)&&isFinite(x.lon)&&x.id&&x.name);}catch(e){}
function saveLM(){try{localStorage.setItem(LMKEY,JSON.stringify(LM));}catch(e){}}
const TMODE={walk:{label:"Walk",srv:"foot",kmh:5},skate:{label:"Skate",srv:"foot",kmh:11,fixed:true},bike:{label:"Bike",srv:"bike",kmh:16},drive:{label:"Drive",srv:"car",kmh:45}};
const routeCache=new Map(),lmLabels=new Map();
const MODE_ICON={
  // next-gen concept hypercar: low wedge, glass canopy, light blade, glowing rims
  drive:`<svg viewBox="0 0 64 40" aria-hidden="true"><defs><linearGradient id="hcB" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#1a1d38"/><stop offset=".55" stop-color="#4a33b8"/><stop offset="1" stop-color="#15173a"/></linearGradient>
    <linearGradient id="hcG" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#bff3ff"/><stop offset="1" stop-color="#2a6fa8"/></linearGradient></defs>
    <ellipse cx="32" cy="34" rx="28" ry="2.4" fill="#000" opacity=".5"/>
    <path d="M4 18.5h9l-1.6 2.6H5.2z" fill="#0e1024" stroke="#6f5cff" stroke-width=".7"/>
    <path d="M3 27c1.5-4.2 8-6 16.5-7.2l9.8-6.4c3.4-1.9 11.4-2 15.2.2l8.2 5.4c5 .9 8.6 3 9.3 6.2l-.5 3.6-58.2.6z" fill="url(#hcB)" stroke="#07081a" stroke-width="1.2" stroke-linejoin="round"/>
    <path d="M27.5 20 33.2 15c3.6-1.3 8.3-1.2 11 .4l5 4.6z" fill="url(#hcG)" stroke="#07081a" stroke-width=".8"/>
    <path d="M34 15.6 33 20" stroke="#07081a" stroke-width=".8"/>
    <path d="M6 25.2h52" stroke="#5ff3ff" stroke-width="1.2" stroke-linecap="round"/>
    <path d="M37 23.3h8.5l-3 2.6h-6.8z" fill="#07081a"/>
    <path d="M57.6 23.4l3.6 1.3" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
    <path d="M3.4 26.6h2.6" stroke="#ff3b6b" stroke-width="1.8" stroke-linecap="round"/>
    <g><circle cx="15.5" cy="29.5" r="5.6" fill="#0b0b12" stroke="#07081a"/><circle cx="15.5" cy="29.5" r="3.4" fill="none" stroke="#5ff3ff" stroke-width="1.3"/><circle cx="15.5" cy="29.5" r="1" fill="#5ff3ff"/></g>
    <g><circle cx="49.5" cy="29.5" r="5.6" fill="#0b0b12" stroke="#07081a"/><circle cx="49.5" cy="29.5" r="3.4" fill="none" stroke="#5ff3ff" stroke-width="1.3"/><circle cx="49.5" cy="29.5" r="1" fill="#5ff3ff"/></g>
    <path d="M22 22.2c6-.8 12-1 17-.8" stroke="#8f7dff" stroke-width=".8" opacity=".8" fill="none"/></svg>`,
  // limited-edition skateboard in a dark-fantasy style: black deck, gold trim, a glowing golden tree inside a ring
  skate:`<svg viewBox="0 0 64 40" aria-hidden="true"><defs><radialGradient id="skG" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#ffd76a" stop-opacity=".55"/><stop offset="1" stop-color="#ffd76a" stop-opacity="0"/></radialGradient>
    <linearGradient id="skD" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2a2217"/><stop offset="1" stop-color="#0d0a06"/></linearGradient></defs>
    <ellipse cx="32" cy="35" rx="24" ry="2.2" fill="#000" opacity=".5"/>
    <g transform="rotate(-9 32 20)">
      <rect x="11" y="23.5" width="7" height="2.6" rx="1" fill="#8d8f96"/><rect x="45" y="23.5" width="7" height="2.6" rx="1" fill="#8d8f96"/>
      <circle cx="11.5" cy="28" r="3.2" fill="#f0b43c" stroke="#2a1a05" stroke-width="1"/><circle cx="18" cy="28" r="3.2" fill="#f0b43c" stroke="#2a1a05" stroke-width="1"/>
      <circle cx="45.5" cy="28" r="3.2" fill="#f0b43c" stroke="#2a1a05" stroke-width="1"/><circle cx="52" cy="28" r="3.2" fill="#f0b43c" stroke="#2a1a05" stroke-width="1"/>
      <rect x="3" y="12" width="58" height="12.5" rx="6.25" fill="url(#skD)" stroke="#1a1206" stroke-width="1.6"/>
      <rect x="4.6" y="13.5" width="54.8" height="9.5" rx="4.75" fill="none" stroke="#d9a93a" stroke-width="1"/>
      <circle cx="32" cy="18.2" r="8" fill="url(#skG)"/>
      <circle cx="32" cy="18.2" r="4.3" fill="none" stroke="#f3c552" stroke-width=".8"/>
      <path d="M32 22.6v-7.2M32 18.4l-2.6-2.6M32 17.4l2.6-2.8M32 20l-3.6-1.2M32 20l3.6-1.3M32 22.6l-2 .6M32 22.6l2 .6" stroke="#ffe08a" stroke-width=".9" stroke-linecap="round"/>
      <path d="M12 18.2h6.5M45.5 18.2H52" stroke="#d9a93a" stroke-width=".7" stroke-dasharray="1 1.4"/>
      <path d="M8 14.5c2-1 5-1.3 8-1.2" stroke="#fff" stroke-width=".9" stroke-linecap="round" opacity=".35"/>
    </g></svg>`,
  // a Puerto Rican kid in goth / alt clothes: black hoodie, chains, silver hoops and rings, ripped black jeans, black work shoes
  walk:`<svg viewBox="0 0 48 64" aria-hidden="true">
    <ellipse cx="24" cy="61" rx="13" ry="2" fill="#000" opacity=".5"/>
    <path d="M16.5 58.5h8.2c1 0 1.4 1.8.4 2.2h-9.5c-.8 0-.6-2.2.9-2.2z" fill="#111" stroke="#000" stroke-width=".6"/>
    <path d="M26 57.3l7.6-.3c1 0 1.5 1.8.5 2.3l-8.6.3c-.9 0-.9-2.2.5-2.3z" fill="#111" stroke="#000" stroke-width=".6"/>
    <path d="M16.5 59.5h9M26 58.5l8.6-.3" stroke="#3a3a3a" stroke-width=".7"/>
    <path d="M17 41l-.8 17.8h8L25 45l2 12.4 7-.6-2.8-16.3z" fill="#16161a" stroke="#000" stroke-width=".7"/>
    <path d="M18.8 50.5l3-.6M28.8 49.6l2.4-.3" stroke="#b07a4e" stroke-width="1.1" stroke-linecap="round"/>
    <path d="M17.6 41.5c4 1.2 9.6 1.2 13.6-.5" stroke="#9aa0a8" stroke-width=".9" fill="none" stroke-dasharray="1.2 .8"/>
    <path d="M14.5 26c0-4 4.3-6 9.5-6s9.8 2 9.8 6.2l.5 16.4c-6 1.8-14 1.8-20.2 0z" fill="#1d1b22" stroke="#000" stroke-width=".8"/>
    <path d="M19.5 21.5c1.4 3 7.8 3 9.4 0" fill="none" stroke="#000" stroke-width=".8"/>
    <path d="M20.5 22.5c.6 4 6 6 7.6 0" fill="none" stroke="#c9ced6" stroke-width=".9"/>
    <circle cx="24.3" cy="26.6" r=".9" fill="#c9ced6"/>
    <path d="M21 30l2 3.5 2.4-3.5" stroke="#6b3fa0" stroke-width="1" fill="none"/>
    <path d="M14.8 26.5c-2.4 3.6-3.4 8-3 12.2l2.6.4c.2-3.6 1.2-6.6 2.6-9" fill="#1d1b22" stroke="#000" stroke-width=".8"/>
    <path d="M33.6 26.5c2.2 3.4 3.4 7.6 3.2 11.8l-2.6.3c-.2-3.4-1-6.2-2.4-8.6" fill="#1d1b22" stroke="#000" stroke-width=".8"/>
    <circle cx="12.8" cy="40.4" r="1.9" fill="#b07a4e" stroke="#5a3a1f" stroke-width=".5"/><circle cx="35.6" cy="39.9" r="1.9" fill="#b07a4e" stroke="#5a3a1f" stroke-width=".5"/>
    <path d="M11.6 39.8h2.4M34.4 39.4h2.4" stroke="#dfe3ea" stroke-width=".8"/>
    <rect x="21.8" y="17.5" width="5" height="4" fill="#a8703f"/>
    <ellipse cx="24.3" cy="12.2" rx="7.4" ry="7.8" fill="#b07a4e" stroke="#5a3a1f" stroke-width=".7"/>
    <path d="M16.6 11.5c-1-5.6 3-9.2 7.6-9.2 5 0 8.6 3.4 7.8 9-1-2-2.4-3-4.2-3.4-1.6 1.2-4.4 1.4-6.6.6-1.6.6-3.2 1.6-4.6 3z" fill="#141013" stroke="#000" stroke-width=".6"/>
    <circle cx="18" cy="5.6" r="2.1" fill="#141013"/><circle cx="21.2" cy="3.4" r="2.2" fill="#141013"/><circle cx="25" cy="2.6" r="2.3" fill="#141013"/><circle cx="28.8" cy="3.6" r="2.2" fill="#141013"/><circle cx="31" cy="6.6" r="2" fill="#141013"/>
    <path d="M20.3 12.4h2.3M26.2 12.4h2.3" stroke="#000" stroke-width="1.4" stroke-linecap="round"/>
    <path d="M20 11.3l3-.6M26 10.7l3 .6" stroke="#000" stroke-width=".7" stroke-linecap="round"/>
    <path d="M22.4 16.3c1.2.8 2.6.8 3.8 0" stroke="#5a2a1a" stroke-width=".8" fill="none" stroke-linecap="round"/>
    <circle cx="16.9" cy="14.8" r="1.3" fill="none" stroke="#dfe3ea" stroke-width=".7"/><circle cx="31.7" cy="14.8" r="1.3" fill="none" stroke="#dfe3ea" stroke-width=".7"/>
    <circle cx="26.9" cy="15.6" r=".45" fill="#dfe3ea"/></svg>`,
  bike:`<svg viewBox="0 0 64 40" aria-hidden="true"><ellipse cx="32" cy="36" rx="26" ry="2" fill="#000" opacity=".5"/>
    <circle cx="15" cy="26" r="8.5" fill="none" stroke="#111" stroke-width="3"/><circle cx="49" cy="26" r="8.5" fill="none" stroke="#111" stroke-width="3"/>
    <circle cx="15" cy="26" r="8.5" fill="none" stroke="#7fd3ff" stroke-width="1.2"/><circle cx="49" cy="26" r="8.5" fill="none" stroke="#7fd3ff" stroke-width="1.2"/>
    <path d="M15 26 25 14h17l7 12M25 14l7 12h-17M32 26 39 9" stroke="#ff5a7a" stroke-width="2.6" fill="none" stroke-linejoin="round" stroke-linecap="round"/>
    <path d="M22 13h7M36 9h6" stroke="#111" stroke-width="2.6" stroke-linecap="round"/><circle cx="32" cy="26" r="2" fill="#111"/></svg>`};
let route=null,routeBlocked=false;const RT={tab:"routes",from:"me",to:"",mode:"walk",origin:"me"};
const r2d=180/Math.PI;
function placeName(){if(!place||place.state!=="done")return"";const c=place.list[place.sel]||place.B;return c&&c.tags&&c.tags.name||"";}
function ptOf(id){
  if(id==="me")return{id,name:ME.name,lat:ME.lat*r2d,lon:ME.lon*r2d};
  if(id==="spot")return place?{id,name:placeName()||"Clicked spot",lat:place.lat,lon:place.lon}:null;
  return LM.find(x=>x.id===id)||null;}
function gcDist(a,b){const p1=a.lat/r2d,p2=b.lat/r2d,dp=p2-p1,dl=(b.lon-a.lon)/r2d,h=Math.sin(dp/2)**2+Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)**2;return 2*6371000*Math.asin(Math.min(1,Math.sqrt(h)));}
function osrm(a,b,srv){
  const c=`${a.lon.toFixed(6)},${a.lat.toFixed(6)};${b.lon.toFixed(6)},${b.lat.toFixed(6)}`,k=srv+":"+c;
  if(routeCache.has(k))return routeCache.get(k);
  const urls=[`https://routing.openstreetmap.de/routed-${srv}/route/v1/driving/${c}?overview=full&geometries=geojson&alternatives=3`];
  if(srv==="car")urls.push(`https://router.project-osrm.org/route/v1/driving/${c}?overview=full&geometries=geojson&alternatives=3`);
  const p=(async()=>{let err;for(const u of urls){try{const j=await fetchJSON(u,15000);
      if(j.code==="Ok"&&j.routes&&j.routes[0])return{alts:j.routes.map(R=>({dist:R.distance,dur:R.duration,geom:R.geometry.coordinates}))};
      if(j.code==="NoRoute")return{none:true};err=new Error(j.code);}catch(e){err=e;}}throw err;})();
  routeCache.set(k,p);p.catch(()=>routeCache.delete(k));return p;}
async function tripTime(a,b,mode,light){const M=TMODE[mode];
  try{if(mode==="skate")return await skateTrip(a,b,light);
    const r=await osrm(a,b,M.srv);if(r.none)return{none:true};
    const alts=r.alts.map(x=>({dist:x.dist,dur:M.fixed?x.dist/1000/M.kmh*3600:x.dur,geom:x.geom})).sort((x,y)=>x.dur-y.dur);   // fastest first
    return{...alts[0],alts,road:true};}
  catch(e){routeBlocked=true;const d=gcDist(a,b)*1.3,x={dist:d,dur:d/1000/M.kmh*3600,geom:[[a.lon,a.lat],[b.lon,b.lat]]};return{...x,alts:[x],road:false};}}
function fmtDur(s){if(!isFinite(s))return"—";const m=Math.max(1,Math.round(s/60));if(m<60)return m+" min";const h=Math.floor(m/60),r=m%60;return h+" h"+(r?" "+r+" min":"");}
function fmtMi(m){const mi=m/1609.344;return mi<.1?Math.round(m*3.28084)+" ft":(mi<10?mi.toFixed(1):Math.round(mi).toLocaleString())+" mi";}
function fitGeom(g){let s=90,n=-90,w=180,e=-180;for(const[lo,la]of g){s=Math.min(s,la);n=Math.max(n,la);w=Math.min(w,lo);e=Math.max(e,lo);}
  const la=(s+n)/2,lo=(w+e)/2,ext=Math.max((n-s)*110.57,(e-w)*111.32*Math.cos(la/r2d),.3),rk=clamp(ext*1.9,.8,6371*3.2);
  const wide=Math.min(rk*3.2+40,6371*3.2);
  goEarthPlace(la/r2d,lo/r2d,wide*KM,8/r2d,()=>{if(route)route.t0=performance.now();goEarthPlace(la/r2d,lo/r2d,rk*KM,.5/r2d);});}   // end straight down, north up: a map
async function plotRoute(){
  const a=ptOf(RT.from),b=ptOf(RT.to);if(!a||!b)return;
  if(a.lat===b.lat&&a.lon===b.lon){route={state:"same",a,b};renderRoutes();return;}
  if(innerWidth<=900&&!$("routes").hidden)toggleRoutes();   // small screens: get the sheet out of the way of the map
  const mode=RT.mode,tok={};route={state:"loading",a,b,mode,tok,others:{}};renderRoutes();
  const r=await tripTime(a,b,mode);if(!route||route.tok!==tok)return;
  Object.assign(route,r,{state:r.none?"none":"done",sel:0,t0:1e15});renderRoutes();
  if(!r.none)fitGeom(r.geom);
  for(const m of Object.keys(TMODE))if(m!==mode)tripTime(a,b,m).then(x=>{if(route&&route.tok===tok){route.others[m]=x;renderRoutes();}});
}
function pickAlt(i){if(!route||!route.alts||!route.alts[i])return;route.sel=i;Object.assign(route,{dist:route.alts[i].dist,dur:route.alts[i].dur,geom:route.alts[i].geom,prof:route.alts[i].prof||null,hover:null});renderRoutes();}
function clearRoute(){route=null;renderRoutes();}
$("routeBtn").addEventListener("click",e=>{e.preventDefault();e.stopPropagation();closeMenus();openRoutes("routes");});
$("routePop").addEventListener("pointermove",e=>{const c=e.target.closest(".skc");if(c)skChartHover(c,e);});
$("routePop").addEventListener("pointerleave",()=>{const c=$("routePop").querySelector(".skc");if(c)skChartLeave(c);});
$("routePop").addEventListener("click",e=>{const a=e.target.closest("[data-alt]");if(a){pickAlt(+a.dataset.alt);return;}
  const m=e.target.closest("[data-mode]");if(m){RT.mode=m.dataset.mode;plotRoute();return;}if(e.target.closest("[data-clear]"))clearRoute();});
function addLandmark(name,lat,lon){const id="l"+Date.now().toString(36)+Math.random().toString(36).slice(2,5);
  LM.push({id,name:String(name).slice(0,80),lat:+lat,lon:+lon});saveLM();renderRoutes();return id;}
function removeLandmark(id){LM=LM.filter(x=>x.id!==id);if(RT.from===id)RT.from="me";if(RT.to===id)RT.to="";if(RT.origin===id)RT.origin="me";
  const el=lmLabels.get(id);if(el){el.remove();lmLabels.delete(id);}saveLM();renderRoutes();}
function openRoutes(tab){if(tab)RT.tab=tab;$("routes").hidden=false;$("tab-places").setAttribute("aria-expanded","true");renderRoutes();}
function toggleRoutes(){const r=$("routes");if(r.hidden)openRoutes();else{r.hidden=true;$("tab-places").setAttribute("aria-expanded","false");}}
function optList(sel,withNone){const o=[`<option value="me"${sel==="me"?" selected":""}>${esc(ME.name)} · my location</option>`];
  if(place)o.push(`<option value="spot"${sel==="spot"?" selected":""}>${esc(placeName()||"Clicked spot")} · last click</option>`);
  LM.forEach(l=>o.push(`<option value="${l.id}"${sel===l.id?" selected":""}>${esc(l.name)}</option>`));
  if(withNone)o.unshift(`<option value=""${sel?"":" selected"}>Choose a destination</option>`);return o.join("");}
const lmTimes=new Map();   // origin|id -> {walk,skate,drive}
function timesFor(o,l){const k=o.id+"|"+l.id+"|"+o.lat+","+o.lon;if(lmTimes.has(k))return lmTimes.get(k);
  const v={};lmTimes.set(k,v);
  ["walk","skate","drive"].forEach(m=>tripTime(o,l,m,true).then(x=>{v[m]=x;if(!$("routes").hidden&&RT.tab==="landmarks")renderRoutes();}));return v;}
function renderPop(){
  const el=$("routePop");if(!route||route.state==="same"){el.hidden=true;return;}el.hidden=false;
  if(route.state==="loading"){el.innerHTML=`<div class="rph"><i class="spin"></i><span>Finding ${TMODE[route.mode].label.toLowerCase()} routes to <b>${esc(route.b.name)}</b></span></div>`;return;}
  if(route.state==="none"){el.innerHTML=`<div class="rph"><span>No ${TMODE[route.mode].label.toLowerCase()} route to <b>${esc(route.b.name)}</b></span><button type="button" class="rpx" data-clear aria-label="Clear route">×</button></div>`;return;}
  const best=route.alts[0].dur;
  const alts=route.alts.map((x,i)=>`<button type="button" class="rpa" data-alt="${i}" aria-pressed="${i===route.sel}"><em>${i===0?"Fastest":"Route "+(i+1)}</em><b>${fmtDur(x.dur)}</b><span>${fmtMi(x.dist)}${i?` · +${fmtDur(x.dur-best)}`:""}</span></button>`).join("");
  const modes=Object.keys(TMODE).map(m=>[m,m===route.mode?route.alts[0]:route.others[m]]).sort((x,y)=>(x[1]&&!x[1].none?x[1].dur:1e12)-(y[1]&&!y[1].none?y[1].dur:1e12))
    .map(([m,x])=>`<button type="button" class="rpm" data-mode="${m}" aria-pressed="${m===route.mode}">${TMODE[m].label} <b>${x?x.none?"—":fmtDur(x.dur):"…"}</b></button>`).join("");
  el.innerHTML=`<div class="rph"><span>${esc(route.a.name)} <i>→</i> <b>${esc(route.b.name)}</b></span><button type="button" class="rpx" data-clear aria-label="Clear route">×</button></div>
    <div class="rpas">${alts}</div>${route.mode==="skate"&&route.prof?skBreakdown(route.prof)+skChart(route.prof):route.mode==="skate"&&route.road?`<div class="rpn">Couldn't load hill data, so this uses a flat 7 mph.</div>`:""}<div class="rpms">${modes}</div>${route.road?"":`<div class="rpn">Straight-line estimate · real routes need the downloaded copy</div>`}`;
}
function renderRoutes(){
  renderPop();
  const body=$("rtBody");if(!body||$("routes").hidden)return;
  document.querySelectorAll("#routes [data-tab]").forEach(b=>b.setAttribute("aria-selected",String(b.dataset.tab===RT.tab)));
  $("lmAdd").hidden=RT.tab!=="landmarks";
  const note=routeBlocked?`<p class="rn">Road routing can't be reached from here (the Claude viewer blocks it), so times are estimates from straight-line distance. The downloaded copy uses real roads and paths.</p>`:"";
  if(RT.tab==="routes"){
    const modes=Object.entries(TMODE).map(([k,M])=>`<button type="button" class="chip" data-mode="${k}" aria-pressed="${RT.mode===k}">${M.label}</button>`).join("");
    let res="";
    if(route){
      if(route.state==="loading")res=`<div class="rr"><div class="rk"><i class="spin"></i>Finding the ${TMODE[route.mode].label.toLowerCase()} route</div></div>`;
      else if(route.state==="same")res=`<div class="rr"><div class="rk">Pick two different places</div></div>`;
      else if(route.state==="none")res=`<div class="rr"><div class="rk">No ${TMODE[route.mode].label.toLowerCase()} route found</div><p class="rn">The routing service found no way between these two places for this mode.</p></div>`;
      else{const oth=Object.keys(TMODE).filter(m=>m!==route.mode).map(m=>{const x=route.others[m];return`<button type="button" class="om" data-mode="${m}"><span>${TMODE[m].label}</span><b>${x?x.none?"No route":fmtDur(x.dur):"…"}</b></button>`;}).join("");
        res=`<div class="rr"><div class="rk">${TMODE[route.mode].label}${route.road?"":" · estimate"}</div><div class="rt"><b>${fmtDur(route.dur)}</b><span>${fmtMi(route.dist)}</span></div>
          <div class="rw">${esc(route.a.name)} <i>→</i> ${esc(route.b.name)}</div>${route.mode==="skate"&&route.prof?skBreakdown(route.prof):""}<div class="oms">${oth}</div>
          <div class="ra"><a class="pb" target="_blank" rel="noopener" href="https://www.google.com/maps/dir/?api=1&origin=${route.a.lat},${route.a.lon}&destination=${route.b.lat},${route.b.lon}&travelmode=${{walk:"walking",skate:"walking",bike:"bicycling",drive:"driving"}[route.mode]}">Open in Google Maps ↗</a><button type="button" class="pb" data-clear>Clear route</button></div></div>`;}
    }
    body.innerHTML=`<label class="fl" for="rtFrom">From</label><div class="sel"><select id="rtFrom">${optList(RT.from)}</select></div>
      <button type="button" class="swap" data-swap aria-label="Swap start and destination">⇅</button>
      <label class="fl" for="rtTo">To</label><div class="sel"><select id="rtTo">${optList(RT.to,true)}</select></div>
      <div class="chips" role="group" aria-label="Travel mode">${modes}</div>
      <button type="button" class="pb pri wide" data-plot${RT.to?"":" disabled"}>Plot route</button>${res}${note}
      <p class="rn">Skate mode reads the hills: downhill you coast (10–15 mph), flat you push (7 mph), gentle climbs slow you down, and anything steeper than 5% you walk. It compares walking and bike routes and picks the fastest to skate.</p>`;
  }else{
    const o=ptOf(RT.origin)||ptOf("me");
    const rows=LM.filter(l=>l.id!==o.id).map(l=>{const v=timesFor(o,l),f=m=>v[m]?v[m].none?"—":fmtDur(v[m].dur):"…";
      return`<li class="lm"><div class="lmh"><b>${esc(l.name)}</b><span>${v.walk&&!v.walk.none?fmtMi(v.walk.dist):""}</span></div>
        <div class="lmt"><span>Walk <b>${f("walk")}</b></span><span>Skate <b>${f("skate")}</b></span><span>Drive <b>${f("drive")}</b></span></div>
        <div class="lma"><button type="button" data-fly="${l.id}">Fly there</button><button type="button" data-route="${l.id}">Route</button><button type="button" data-del="${l.id}" aria-label="Remove ${esc(l.name)}">Remove</button></div></li>`;}).join("");
    body.innerHTML=`<label class="fl" for="rtOrigin">Times from</label><div class="sel"><select id="rtOrigin">${optList(RT.origin)}</select></div>
      ${LM.length?`<ul class="lms">${rows||`<li class="empty">This is your only landmark. Add another to compare times.</li>`}</ul>`:`<p class="empty">No landmarks yet. Search for a place below, or click a building up close and choose <b>Save as landmark</b>.</p>`}${note}`;
  }
}
$("routes").addEventListener("click",e=>{
  const t=e.target.closest("[data-tab]");if(t){RT.tab=t.dataset.tab;renderRoutes();return;}
  const m=e.target.closest("[data-mode]");if(m){RT.mode=m.dataset.mode;if(route&&route.state!=="loading"&&RT.to)plotRoute();else renderRoutes();return;}
  if(e.target.closest("[data-swap]")){const f=RT.from;RT.from=RT.to||"me";RT.to=f;renderRoutes();return;}
  if(e.target.closest("[data-plot]")){plotRoute();return;}
  if(e.target.closest("[data-clear]")){clearRoute();return;}
  if(e.target.closest("#routesX")){toggleRoutes();return;}
  const f=e.target.closest("[data-fly]");if(f){const l=ptOf(f.dataset.fly);if(l)flyToPlace({name:l.name,detail:"",lat:l.lat,lon:l.lon,box:null,src:"Landmark"});return;}
  const r=e.target.closest("[data-route]");if(r){RT.from=RT.origin;RT.to=r.dataset.route;RT.tab="routes";plotRoute();return;}
  const d=e.target.closest("[data-del]");if(d){removeLandmark(d.dataset.del);}
});
$("routes").addEventListener("change",e=>{const id=e.target.id;if(id==="rtFrom")RT.from=e.target.value;else if(id==="rtTo")RT.to=e.target.value;else if(id==="rtOrigin")RT.origin=e.target.value;else return;renderRoutes();});
$("lmAdd").addEventListener("submit",async e=>{e.preventDefault();const inp=$("lmQ"),q=inp.value.trim(),msg=$("lmMsg");if(!q)return;msg.textContent="Finding "+q+"…";
  try{const r=await geocode(q);if(!r.length){msg.textContent="No place found. Try adding a town or state.";return;}
    const x=r[0];addLandmark(x.name,x.lat,x.lon);inp.value="";msg.textContent=`Saved ${x.name}${x.detail?" · "+x.detail:""}`;}
  catch(err){msg.textContent=err.message||"The search failed.";}});
function drawRoute(rt,up,fw){
  const p=$("routePath"),a=$("routeA"),b=$("routeB");
  if(!route||!route.geom||route.state!=="done"||len(sub(cp,EW))>13*6){$("routeBadge").hidden=true;p.setAttribute("d","");$("routeAlt").setAttribute("d","");a.setAttribute("visibility","hidden");b.setAttribute("visibility","hidden");drawSkate(rt,up,fw,null,0);return;}
  const g=route.geom,step=Math.max(1,Math.floor(g.length/1500));let d="",pen=false;
  const vis=(la,lo)=>{const pw=add(EW,qApply(EQ,mul(efDir(la/r2d,lo/r2d),13)));return dot(sub(pw,EW),sub(cp,pw))>0?project(pw,rt,up,fw):null;};
  for(let i=0;i<g.length;i+=step){const q=vis(g[i][1],g[i][0]);if(!q){pen=false;continue;}d+=(pen?"L":"M")+q[0].toFixed(1)+" "+q[1].toFixed(1);pen=true;}
  {const q=vis(g[g.length-1][1],g[g.length-1][0]);if(q&&pen)d+="L"+q[0].toFixed(1)+" "+q[1].toFixed(1);}
  const k=clamp((performance.now()-route.t0)/1600,0,1),e=1-Math.pow(1-k,3);
  p.setAttribute("d",d);p.classList.toggle("est",!route.road);p.style.strokeDasharray=k<1?"1 0":"";
  if(k<1){const L=p.getTotalLength?p.getTotalLength():0;p.style.strokeDasharray=`${L*e} ${L}`;}
  let da="";route.alts.forEach((x,i)=>{if(i===route.sel)return;let pen2=false;const st=Math.max(1,Math.floor(x.geom.length/600));
    for(let j=0;j<x.geom.length;j+=st){const q=vis(x.geom[j][1],x.geom[j][0]);if(!q){pen2=false;continue;}da+=(pen2?"L":"M")+q[0].toFixed(1)+" "+q[1].toFixed(1);pen2=true;}});
  $("routeAlt").setAttribute("d",k>.6?da:"");
  if(drawSkate(rt,up,fw,vis,k))p.setAttribute("d","");   // skate mode draws the route by terrain instead
  // once traced, a badge at the route's midpoint shows how you'll travel
  const bd=$("routeBadge"),mid=g[Math.floor(g.length/2)],qm=k>=1&&mid?vis(mid[1],mid[0]):null;
  if(qm){if(bd.dataset.m!==route.mode){bd.dataset.m=route.mode;bd.innerHTML=`<div>${MODE_ICON[route.mode]}</div>`;bd.title=TMODE[route.mode].label;}bd.hidden=false;bd.style.transform=`translate(${(qm[0]-15).toFixed(1)}px,${(qm[1]-8).toFixed(1)}px)`;}else bd.hidden=true;
  [[a,g[0]],[b,g[g.length-1]]].forEach(([el,c])=>{const q=vis(c[1],c[0]);if(q){el.setAttribute("cx",q[0].toFixed(1));el.setAttribute("cy",q[1].toFixed(1));el.removeAttribute("visibility");}else el.setAttribute("visibility","hidden");});
}
function drawLandmarks(rt,up,fw){
  const near=OPT.me&&len(sub(cp,EW))<13*40;
  LM.forEach(l=>{let el=lmLabels.get(l.id);if(!el){el=mkLabel(esc(l.name),"lm");lmLabels.set(l.id,el);}
    if(!near){el.hidden=true;return;}const pw=add(EW,qApply(EQ,mul(efDir(l.lat/r2d,l.lon/r2d),13))),q=project(pw,rt,up,fw);
    if(q&&dot(sub(pw,EW),sub(cp,pw))>0){el.hidden=false;el.style.transform=`translate(${(q[0]-5).toFixed(1)}px,${(q[1]-5).toFixed(1)}px)`;}else el.hidden=true;});
}

// ---------- skate mode: terrain-aware routing ----------
// Elevation along the route (Open-Meteo, Copernicus 90 m DEM) splits it into stretches you coast, push, grind up, or walk.
const SKC={coast:{label:"Coast",col:"#22a85e"},push:{label:"Push",col:"#4192d0"},grind:{label:"Grind uphill",col:"#bb8418"},walk:{label:"Walk",col:"#d6479a"}};
const SKN=["coast","push","grind","walk"];
const elevCache=new Map();
async function elevations(pts){
  const out=new Array(pts.length),jobs=[];
  for(let i=0;i<pts.length;i+=100){const s=pts.slice(i,i+100),la=s.map(p=>p[0].toFixed(5)).join(","),lo=s.map(p=>p[1].toFixed(5)).join(","),k=la+"|"+lo;
    jobs.push((async()=>{let v=elevCache.get(k);if(!v){const j=await fetchJSON(`https://api.open-meteo.com/v1/elevation?latitude=${la}&longitude=${lo}`,12000);v=j&&j.elevation;
      if(!Array.isArray(v)||v.length!==s.length)throw new Error("elevation");elevCache.set(k,v);}v.forEach((e,j)=>out[i+j]=+e);})());}
  await Promise.all(jobs);return out;}
function resample(g,maxN){   // evenly spaced points along a GeoJSON line, at least 25 m apart
  const d=[0];for(let i=1;i<g.length;i++)d.push(d[i-1]+gcDist({lat:g[i-1][1],lon:g[i-1][0]},{lat:g[i][1],lon:g[i][0]}));
  const L=d[d.length-1],n=Math.max(2,Math.min(maxN,Math.round(L/25))+1),pts=[];let j=0;
  for(let k=0;k<n;k++){const s=L*k/(n-1);while(j<g.length-2&&d[j+1]<s)j++;const f=clamp((s-d[j])/Math.max(1e-9,d[j+1]-d[j]),0,1);
    pts.push({s,lat:g[j][1]+(g[j+1][1]-g[j][1])*f,lon:g[j][0]+(g[j+1][0]-g[j][0])*f});}
  return{pts,L};}
const skSpeed=(c,gr)=>c==="coast"?clamp(16+(-gr-.01)*400,16,24):c==="push"?11:c==="grind"?clamp(11-(gr-.015)/.035*5,6,11):4.5;   // km/h
function skateProfile(raw,pts){
  const n=pts.length,step=(pts[n-1].s-pts[0].s)/Math.max(1,n-1),w=Math.max(1,Math.round(40/Math.max(step,1)));
  const e=raw.map((_,i)=>{let s=0,c=0;for(let k=Math.max(0,i-w);k<=Math.min(n-1,i+w);k++){s+=raw[k];c++;}return s/c;});   // DEM noise off
  const gr=[],cl=[];
  for(let i=0;i<n-1;i++){const a=Math.max(0,i-w),b=Math.min(n-1,i+1+w),g=(e[b]-e[a])/Math.max(1,pts[b].s-pts[a].s);gr.push(g);
    cl.push(g<=-.01?"coast":g<.015?"push":g<.05?"grind":"walk");}
  // stretches shorter than 60 m take their neighbour's class (a driveway dip isn't a coast)
  let runs=[];for(let i=0;i<cl.length;i++){const r=runs[runs.length-1];if(r&&r.c===cl[i])r.i1=i;else runs.push({c:cl[i],i0:i,i1:i});}
  const rl=r=>pts[r.i1+1].s-pts[r.i0].s;
  for(let pass=0;pass<3;pass++){runs.forEach((r,k)=>{if(rl(r)<60&&runs.length>1){const nb=runs[k-1]||runs[k+1];for(let i=r.i0;i<=r.i1;i++)cl[i]=nb.c;}});
    runs=[];for(let i=0;i<cl.length;i++){const r=runs[runs.length-1];if(r&&r.c===cl[i])r.i1=i;else runs.push({c:cl[i],i0:i,i1:i});}}
  const by={coast:0,push:0,grind:0,walk:0};let time=0,gain=0,loss=0;
  for(let i=0;i<cl.length;i++){const L=pts[i+1].s-pts[i].s;by[cl[i]]+=L;time+=L/1000/skSpeed(cl[i],gr[i])*3600;const de=e[i+1]-e[i];if(de>0)gain+=de;else loss-=de;}
  const marks=[];
  runs.forEach(r=>{const L=rl(r);let mx=0,mn=0,avg=(e[r.i1+1]-e[r.i0])/Math.max(1,L);for(let i=r.i0;i<=r.i1;i++){mx=Math.max(mx,gr[i]);mn=Math.min(mn,gr[i]);}
    if(r.c==="walk"&&L>=30)marks.push({k:"walk",i:r.i0,L,g:mx,rise:e[r.i1+1]-e[r.i0]});
    if(r.c==="coast"&&L>=90)marks.push({k:mn<-.08?"steep":"coast",i:r.i0,L,g:avg,g2:mn});});
  const pick=(k,n)=>marks.filter(m=>m.k===k).sort((a,b)=>b.L-a.L).slice(0,n);
  const shown=[...pick("walk",8),...pick("steep",4),...pick("coast",6)].sort((a,b)=>a.i-b.i);
  return{pts,e,gr,cl,runs,by,time,gain,loss,marks:shown,hills:marks.filter(m=>m.k==="walk").length,minE:Math.min(...e),maxE:Math.max(...e),L:pts[n-1].s};
}
async function skateTrip(a,b,light){
  const [f,bk]=await Promise.all([osrm(a,b,"foot").catch(e=>({err:e})),light?null:osrm(a,b,"bike").catch(()=>null)]);
  if(f&&f.err&&!(bk&&bk.alts))throw f.err;
  let cands=[];[f,bk].forEach(r=>{if(r&&r.alts)cands.push(...r.alts);});
  if(!cands.length)return{none:true};
  cands=cands.filter((c,i)=>!cands.slice(0,i).some(o=>Math.abs(o.dist-c.dist)<o.dist*.015&&gcDist({lat:o.geom[o.geom.length>>1][1],lon:o.geom[o.geom.length>>1][0]},{lat:c.geom[c.geom.length>>1][1],lon:c.geom[c.geom.length>>1][0]})<60));
  const res=await Promise.all((light?cands.slice(0,1):cands.slice(0,4)).map(async c=>{
    try{const{pts}=resample(c.geom,300),el=await elevations(pts.map(p=>[p.lat,p.lon])),pr=skateProfile(el,pts);return{dist:c.dist,dur:pr.time,geom:c.geom,prof:pr};}
    catch(e){return{dist:c.dist,dur:c.dist/1000/11*3600,geom:c.geom,prof:null};}}));
  res.sort((x,y)=>x.dur-y.dur);
  return{...res[0],alts:res,road:true};
}
const fmtFt=m=>Math.round(m*3.28084).toLocaleString()+" ft";
const SK_ICON={coast:'<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 4l9 8M12 7v5H7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  walk:'<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><ellipse cx="5" cy="5.5" rx="2" ry="3"/><ellipse cx="11" cy="9.5" rx="2" ry="3"/><circle cx="5" cy="10.4" r="1.1"/><circle cx="11" cy="14.2" r="1.1"/></svg>',
  steep:'<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2 15 14H1z" fill="currentColor"/><path d="M8 6v4M8 12v.4" stroke="#0a100e" stroke-width="1.8" stroke-linecap="round"/></svg>'};
const mph=g=>Math.round(skSpeed("coast",g)/1.609);
function skMarkText(m){return m.k==="walk"?`Walk · ${Math.round(m.g*100)}% hill`:m.k==="steep"?`Steep · slow down`:`Coast · up to ${mph(m.g2)} mph`;}
function skBreakdown(pr){
  return`<div class="skl">${SKN.filter(c=>pr.by[c]>=30).map(c=>`<span class="sk-${c}"><i></i>${SKC[c].label} <b>${fmtMi(pr.by[c])}</b>${c==="walk"&&pr.hills?` (${pr.hills} ${pr.hills===1?"hill":"hills"})`:""}</span>`).join("")}</div>`;}
function skChart(pr){   // elevation profile, one scale, coloured by stretch
  const W=480,H=104,l=46,r=10,t=8,b=20,iw=W-l-r,ih=H-t-b,lo=pr.minE,hi=Math.max(pr.maxE,lo+10),n=pr.pts.length;
  const X=s=>l+s/pr.L*iw,Y=v=>t+(hi-v)/(hi-lo)*ih;
  let fills="",lines="";
  pr.runs.forEach(rn=>{let top="";for(let i=rn.i0;i<=rn.i1+1;i++)top+=(i===rn.i0?"M":"L")+X(pr.pts[i].s).toFixed(1)+" "+Y(pr.e[i]).toFixed(1);
    fills+=`<path d="${top}L${X(pr.pts[rn.i1+1].s).toFixed(1)} ${t+ih}L${X(pr.pts[rn.i0].s).toFixed(1)} ${t+ih}Z" class="skf sk-${rn.c}"/>`;
    lines+=`<path d="${top}" class="skline sk-${rn.c}"/>`;});
  const grid=[hi,lo].map(v=>`<line x1="${l}" x2="${W-r}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="skg"/><text x="${l-6}" y="${(Y(v)+3.5).toFixed(1)}" text-anchor="end" class="skt">${fmtFt(v)}</text>`).join("");
  return`<div class="skc"><div class="skh"><span>Hills along the way</span><span>${fmtFt(pr.gain)} up · ${fmtFt(pr.loss)} down</span></div>
    <svg viewBox="0 0 ${W} ${H}" class="skp" role="img" aria-label="Elevation profile from ${fmtFt(pr.e[0])} to ${fmtFt(pr.e[n-1])}, climbing ${fmtFt(pr.gain)} and dropping ${fmtFt(pr.loss)}">
    ${grid}${fills}${lines}<text x="${l}" y="${H-5}" class="skt">0</text><text x="${W-r}" y="${H-5}" text-anchor="end" class="skt">${fmtMi(pr.L)}</text>
    <line class="skx" x1="0" x2="0" y1="${t}" y2="${t+ih}" visibility="hidden"/><circle class="skd" r="4" visibility="hidden"/>
    <rect x="${l}" y="${t}" width="${iw}" height="${ih}" fill="transparent" class="skhit"/></svg><div class="sktip" hidden></div></div>`;}
function skChartHover(el,e){   // crosshair + tooltip; the matching point lights up on the map
  const pr=route&&route.prof;if(!pr)return;const svg=el.querySelector(".skp"),rc=svg.getBoundingClientRect(),W=480,l=46,r=10,t=8,ih=104-8-20,iw=W-l-r;
  const x=(e.clientX-rc.left)/rc.width*W,s=clamp((x-l)/iw,0,1)*pr.L;let i=0;while(i<pr.pts.length-2&&pr.pts[i+1].s<s)i++;
  const hi=Math.max(pr.maxE,pr.minE+10),Y=v=>t+(hi-v)/(hi-pr.minE)*ih,X=l+pr.pts[i].s/pr.L*iw;
  const ln=svg.querySelector(".skx"),d=svg.querySelector(".skd"),tip=el.querySelector(".sktip");
  ln.setAttribute("x1",X);ln.setAttribute("x2",X);ln.removeAttribute("visibility");d.setAttribute("cx",X);d.setAttribute("cy",Y(pr.e[i]));d.removeAttribute("visibility");
  const c=pr.cl[Math.min(i,pr.cl.length-1)],g=pr.gr[Math.min(i,pr.gr.length-1)];
  tip.hidden=false;tip.innerHTML=`<span class="sk-${c}"><i></i><b>${SKC[c].label}</b></span> · ${fmtMi(pr.pts[i].s)} in · ${fmtFt(pr.e[i])}`;
  tip.style.left=clamp(X/W*100,12,88)+"%";route.hover=i;}
function skChartLeave(el){const svg=el.querySelector(".skp");if(!svg)return;svg.querySelector(".skx").setAttribute("visibility","hidden");svg.querySelector(".skd").setAttribute("visibility","hidden");el.querySelector(".sktip").hidden=true;if(route)route.hover=null;}
const skMarkEls=[];
function drawSkate(rt,up,fw,vis,k){
  const pr=route&&route.state==="done"&&route.prof,svgP={};SKN.forEach(c=>svgP[c]="");let cas="";
  const layer=$("skMarks");
  if(!pr||len(sub(cp,EW))>13*6){SKN.forEach(c=>$("sk-"+c).setAttribute("d",""));$("skCase").setAttribute("d","");skMarkEls.forEach(x=>x.hidden=true);$("skHover").setAttribute("visibility","hidden");return false;}
  const n=pr.pts.length,lim=Math.floor(k*(n-1)),Q=pr.pts.map(p=>vis(p.lat,p.lon));
  let prev=null;
  for(let i=0;i<Math.min(lim,n-1);i++){const a=Q[i],b=Q[i+1];if(!a||!b){prev=null;continue;}const c=pr.cl[i];
    const seg=`M${a[0].toFixed(1)} ${a[1].toFixed(1)}L${b[0].toFixed(1)} ${b[1].toFixed(1)}`;
    svgP[c]+=(prev===c?`L${b[0].toFixed(1)} ${b[1].toFixed(1)}`:seg);prev=c;cas+=seg;}
  SKN.forEach(c=>$("sk-"+c).setAttribute("d",svgP[c]));$("skCase").setAttribute("d",cas);
  // markers where coasting and walking start, once the line has finished tracing
  pr.marks.forEach((m,j)=>{let el=skMarkEls[j];if(!el){el=document.createElement("div");el.className="skm";layer.appendChild(el);skMarkEls[j]=el;}
    const key=route.tok+"|"+route.sel+"|"+j;if(el.dataset.key!==key){el.dataset.key=key;el.className="skm skm-"+m.k;el.innerHTML=`<i>${SK_ICON[m.k]}</i><span>${skMarkText(m)}</span>`;}
    const q=k>=1?Q[m.i]:null;if(q){el.hidden=false;el.style.transform=`translate(${(q[0]-11).toFixed(1)}px,${(q[1]-11).toFixed(1)}px)`;}else el.hidden=true;});
  for(let j=pr.marks.length;j<skMarkEls.length;j++)skMarkEls[j].hidden=true;
  const h=$("skHover"),hq=route.hover!=null?Q[route.hover]:null;if(hq){h.setAttribute("cx",hq[0].toFixed(1));h.setAttribute("cy",hq[1].toFixed(1));h.removeAttribute("visibility");}else h.setAttribute("visibility","hidden");
  return true;
}

// ---------- labels over the scene ----------
const lblLayer=$("labels"),LBL=[];
function mkLabel(name,cls){const d=document.createElement("div");d.className="pl"+(cls?" "+cls:"");d.innerHTML=`<i></i><span>${name}</span><em></em>`;d.hidden=true;lblLayer.appendChild(d);return d;}
const labelFor=[...BODY.map(b=>({el:mkLabel(b.name),pos:()=>bodyPos(BODY.indexOf(b)),r:()=>b.r*KM})),{el:mkLabel("Sun"),pos:()=>S.sun,r:()=>695700*KM},{el:mkLabel("Jarvis core"),pos:()=>[0,0,0],r:()=>36}];
const pinEl=mkLabel("","pin"),meEl=mkLabel(ME.name,"me");
function project(p,rt,up,fw){const v=sub(p,cp),z=dot(v,fw);if(z<=1e-6)return null;const s=Math.min(cv.clientWidth,cv.clientHeight)*1.7/z;return[cv.clientWidth/2+dot(v,rt)*s,cv.clientHeight/2-dot(v,up)*s,z];}
function updateLabels(rt,up,fw){
  const show=OPT.orbits;
  labelFor.forEach(L=>{
    if(!show){L.el.hidden=true;return;}
    const p=L.pos(),q=project(p,rt,up,fw);
    if(!q){L.el.hidden=true;return;}
    const px=L.r()/q[2]*Math.min(cv.clientWidth,cv.clientHeight)*1.7;
    if(px>60||q[0]<-50||q[1]<-20||q[0]>cv.clientWidth+50||q[1]>cv.clientHeight+20){L.el.hidden=true;return;}
    L.el.hidden=false;L.el.style.transform=`translate(${(q[0]-5).toFixed(1)}px,${(q[1]-5).toFixed(1)}px)`;
    L.el.querySelector("em").textContent=fmtDist(len(sub(p,cp))/KM);
  });
  if(pin){const pw=add(EW,qApply(EQ,mul(efDir(pin.lat,pin.lon),13))),q=project(pw,rt,up,fw),vis=q&&dot(sub(pw,EW),sub(cp,pw))>0;
    if(vis&&q[2]<13*1.2){pinEl.hidden=false;pinEl.style.transform=`translate(${(q[0]-6).toFixed(1)}px,${(q[1]-6).toFixed(1)}px)`;pinEl.querySelector("span").textContent=pin.name;pinEl.querySelector("em").textContent="";}
    else pinEl.hidden=true;}else pinEl.hidden=true;
  if(OPT.me&&len(sub(cp,EW))<13*40){const pw=add(EW,qApply(EQ,mul(efDir(ME.lat,ME.lon),13))),q=project(pw,rt,up,fw);
    if(q&&dot(sub(pw,EW),sub(cp,pw))>0){meEl.hidden=false;meEl.style.transform=`translate(${(q[0]-5).toFixed(1)}px,${(q[1]-5).toFixed(1)}px)`;}else meEl.hidden=true;}else meEl.hidden=true;
  drawPlace(rt,up,fw);drawRoute(rt,up,fw);drawLandmarks(rt,up,fw);
}

// ---------- orbit lines ----------
const lineVAO=gl.createVertexArray(),lineBuf=gl.createBuffer();
gl.bindVertexArray(lineVAO);gl.bindBuffer(gl.ARRAY_BUFFER,lineBuf);gl.bufferData(gl.ARRAY_BUFFER,12*260*10,gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);gl.bindVertexArray(vao);
function drawOrbits(rt,up,fw){
  if(!OPT.orbits||!ORB)return;
  const [ni,nd]=nearestBody();const fade=clamp((nd-3)/25,0,1);if(fade<=0)return;
  gl.useProgram(P.line.p);gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,W,H);
  gl.uniformMatrix3fv(P.line.u("VW"),false,new Float32Array([rt[0],up[0],fw[0],rt[1],up[1],fw[1],rt[2],up[2],fw[2]]));
  gl.uniform2f(P.line.u("SC"),2*Math.min(W,H)/W,2*Math.min(W,H)/H);gl.uniform1f(P.line.u("F"),1.7);
  gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE);gl.bindVertexArray(lineVAO);gl.bindBuffer(gl.ARRAY_BUFFER,lineBuf);
  const acc=hexRGB(PAL[OPT.pal].acc);
  for(const o of ORB){
    const base=o.geo?EW:S.sun,arr=new Float32Array(o.pts.length*3);
    o.pts.forEach((p,i)=>{const w=add(base,mul(e2w(p),AUU)),v=sub(w,cp);arr[i*3]=v[0];arr[i*3+1]=v[1];arr[i*3+2]=v[2];});
    gl.bufferSubData(gl.ARRAY_BUFFER,0,arr);
    gl.uniform4f(P.line.u("C"),acc[0],acc[1],acc[2],(o.key==="earth"?.5:.28)*fade);
    gl.drawArrays(gl.LINE_STRIP,0,o.pts.length);
  }
  gl.disable(gl.BLEND);gl.bindVertexArray(vao);
}

// ---------- where you are ----------
function sunEF(real){const s=qInv(EQ,norm(sub(S.sun,EW)));if(real||OPT.sun==="live")return s;return s;}
function readout(){
  const v=sub(cp,EW),dist=len(v),alt=(dist-13)/KM;
  const utc=new Date().toISOString().replace("T"," ").slice(0,16)+" UTC";
  if(dist<13*4){
    const q=norm(qInv(EQ,v)),[la,lo]=efLatLon(q),sE=sunEF(true),el=Math.asin(clamp(dot(q,sE),-1,1))*180/Math.PI;
    const ns=`${Math.abs(la*180/Math.PI).toFixed(4)}°${la>=0?"N":"S"} ${Math.abs(lo*180/Math.PI).toFixed(4)}°${lo>=0?"E":"W"}`;
    const near=pin&&Math.acos(clamp(dot(q,efDir(pin.lat,pin.lon)),-1,1))*6371<Math.max(60,alt*1.5);
    const sol=((new Date().getUTCHours()+new Date().getUTCMinutes()/60+lo*12/Math.PI)%24+24)%24;
    $("ro1").textContent=near?pin.name:"Earth";
    $("ro2").textContent=`${near&&pin.detail?pin.detail+" · ":""}${ns} · alt ${fmtDist(alt)} · sun ${el>=0?"+":""}${el.toFixed(0)}° · solar time ${String(sol|0).padStart(2,"0")}:${String(Math.round(sol%1*60)%60).padStart(2,"0")} · ${utc}`;
  }else{
    const [i,d]=nearestBody(),core=len(cp);
    if(core<400){$("ro1").textContent="Jarvis core · merging binary black hole";$("ro2").textContent=`${fmtDist(core/KM)} from the pair · separation ${fmtDist(BSEP/KM)} · ${utc}`;}
    else{const b=BODY[i];$("ro1").textContent=d<60?b.name:"Deep space";$("ro2").textContent=`${fmtDist((len(sub(cp,bodyPos(i)))-bodyR(i))/KM)} above ${b.name} · ${fmtDist(len(sub(cp,S.sun))/KM)} from the Sun · ${utc}`;}
  }
  $("stSub").textContent=" · "+(travel?"flying":travelName.toUpperCase());
}

// ---------- the frame loop ----------
let prevQ=null;
function pollTimers(){
  while(queries.length){const q=queries[0];if(!gl.getQueryParameter(q,gl.QUERY_RESULT_AVAILABLE))break;
    if(!gl.getParameter(TQ.GPU_DISJOINT_EXT)){const ms=gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6;gpuMs=gpuSeen?gpuMs+(ms-gpuMs)*.2:ms;gpuSeen=true;}
    gl.deleteQuery(q);queries.shift();}
}
const f32=a=>new Float32Array(a);
let ctxLost=false,started=false,slowT=0,blurNoted=false,lastDraw=0,coreOn=true,coreWas=true,filOn=true;
const LADDER=["ultra","auto","perf","lite"];
function frame(nowP){
  if(ctxLost){requestAnimationFrame(frame);return;}
  if(!started){
    if(!programsReady()){$("ro1").textContent="Starting the engine";$("ro2").textContent=`Compiling the graphics on your GPU… ${Math.round(nowP/1000)} s (the first start on Windows can take up to a minute)`;last=nowP;requestAnimationFrame(frame);return;}
    started=true;if(!programsOK()){$("ro1").textContent="This graphics card couldn't build the scene";$("ro2").textContent="Details are in the browser console";return;}
  }
  // parked near the Earth with nothing moving: draw at 30 fps instead of 60 to save power
  const still=!drag&&!travel&&!panV&&Math.abs(flyF)<1e-3&&idleT>1.5&&!lookOn()&&len(sub(cp,EW))-13<13*3;
  if(still&&nowP-lastDraw<30){requestAnimationFrame(frame);return;}
  lastDraw=nowP;
  const dtms=Math.min(100,nowP-last);last=nowP;const dt=dtms/1000;t+=dt;frameN=(frameN+1)%1024;
  if(!still)ema+=(dtms-ema)*.08;
  // if even the lowest resolution of this preset can't hold ~20 fps for a few seconds, drop a level (never below Lite)
  if(ema>50&&ss<=PRESET[OPT.q].min+.01&&frameN>30){slowT+=dt;if(slowT>3){const i=LADDER.indexOf(OPT.q);if(i<LADDER.length-1){OPT.q=LADDER[i+1];ss=PRESET[OPT.q].max;save();syncMenus();toast(`<b>Quality lowered to ${PRESET[OPT.q].label}</b> to keep it running smoothly on this device. Change it under Display.`);}slowT=0;}}else slowT=Math.max(0,slowT-dt);
  const pr=PRESET[OPT.q];adjustT-=dt;ceilT-=dt;if(ceilT<=0)ceiling=pr.max;
  if(TQ)pollTimers();
  if(OPT.q!=="auto"){if(adjustT<=0){adjustT=.25;if(ema>30)ss=Math.max(pr.max*.6,ss*.9);else if(ema<20)ss=Math.min(pr.max,ss+.05);}}
  else if(adjustT<=0){adjustT=.25;
    if(gpuSeen){if(gpuMs>20)ss=Math.max(pr.min,ss*.75);else if(gpuMs>13)ss=Math.max(pr.min,ss-.06);else if(gpuMs<6)ss=Math.min(pr.max,ss+.1);else if(gpuMs<9.5)ss=Math.min(pr.max,ss+.03);}
    else{if(ema>24){ss=Math.max(pr.min,ss*.8);ceiling=ss;ceilT=60;}else if(ema>18.5){ss=Math.max(pr.min,ss-.08);ceiling=ss;ceilT=60;}else if(ema<17.2&&ss+.04<=Math.min(pr.max,ceiling))ss+=.04;}
  }
  // Jarvis's state: spin, energy and voice glide to their new values
  const mode=OPT.mode,tv=mode===2?.35+.65*Math.abs(Math.sin(t*7.3)*Math.sin(t*2.1+1)):0;
  energy+=((mode?1:0)-energy)*Math.min(1,dt*2.5);voice+=(tv-voice)*Math.min(1,dt*10);surge*=Math.exp(-dt*2.2);
  spinV+=(18*(SPIN[mode]-spin)-8.5*spinV)*dt;spin+=spinV*dt;diskT+=dt*spin;
  // the binary: a late inspiral, breathing in and out; orbital speed follows Kepler (period ∝ a^1.5)
  BSEP=3.7+.4*Math.sin(t*TWO_PI/140);BPH+=dt*spin*(TWO_PI/22)*Math.pow(3.7/BSEP,1.5);RINGA+=dt*.012*spin;
  // the drones only fly while the core can be seen; they restart without a stale tail
  if(coreOn){if(!coreWas)LIGHTS.forEach(L=>{L.hn=0;});updateLights(dt,t,.55+.45*spin);}coreWas=coreOn;

  // the sky, right now
  const now=Date.now();astro(now);if(filOn)updateNet();
  // near the Earth the camera turns with it, so the ground stays put under you
  if(prevQ&&!travel&&len(sub(cp,EW))<13*8){
    const d=sub(cp,EW),loc=qInv(prevQ,d);cp=add(EW,qApply(EQ,loc));
    const tf=v=>qApply(EQ,qInv(prevQ,v));F={X:norm(tf(F.X)),U:norm(tf(F.U)),Z:norm(tf(F.Z))};
  }
  prevQ=EQ.map(c=>c.slice());

  // camera
  if(travel)travelStep(dt);
  if(homeLock&&!drag&&!travel)orbitV+=(DRIFT[mode]-orbitV)*Math.min(1,dt*1.2);else if(!drag)orbitV*=Math.exp(-dt*2);
  if(!drag&&!travel&&Math.abs(orbitV)>1e-4&&focusBody()<0)orbitHole(orbitV*dt,0);
  if(!drag&&panV&&!travel){const k=Math.exp(-dt*3.5);panV.dx*=k;panV.dy*=k;if(Math.hypot(panV.dx,panV.dy)<2)panV=null;else grab(panV.dx*dt,panV.dy*dt,panV.body);}
  if(Math.abs(flyF)>1e-4){const f=dirF(ly,lp),sp=flyF*surfaceDist();cp=add(cp,mul(f,sp*dt));flyF*=Math.exp(-dt*2.4);homeLock=false;}
  const kl=1-Math.exp(-dt*12);ly+=angDiff(ly,tly)*kl;lp+=(tlp-lp)*kl;
  keepClear();
  { // level the horizon to the nearest world when you're close to it
    const [i,d]=nearestBody();const Ut=d<.35?norm(sub(cp,bodyPos(i))):d<14?(i===2?EQ[1]:S.bodies[i].pole):[0,1,0];if(!travel||d<14)levelTo(Ut,dt*1.4);
  }
  updateWindows(dt);
  if(!pickHinted&&esriState==="ok"&&pickable()&&WIN[0].live){pickHinted=true;toast(`<b>Click any building</b> to see its name, address, phone number and website.`);}
  if(!blurNoted&&esriState==="blocked"&&len(sub(cp,EW))-13<300*KM){blurNoted=true;toast(`<b>Sharp satellite imagery can't load inside the Claude viewer</b>, which blocks outside map servers. Up close you're seeing NASA's 7 km-per-pixel map. Open the downloaded copy of this page in Edge to stream imagery down to about 0.3 m.`);}
  idleT+=dt;if(OPT.autohide&&idleT>6&&!openMenu&&document.activeElement!==$("q")&&$("results").hidden&&$("place").hidden&&$("routes").hidden)document.body.classList.add("idle");

  const dpr=Math.min(devicePixelRatio||1,2);
  ensure(Math.max(2,Math.round(cv.clientWidth*dpr)),Math.max(2,Math.round(cv.clientHeight*dpr)));
  if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H;}
  const sw=Math.min(sceneT.w,Math.round(W*ss)),sh=Math.min(sceneT.h,Math.round(H*ss));
  const fw=dirF(ly+Math.sin(t*.23)*.0012,clampP(lp+Math.sin(t*.31)*.0008));
  const rt=norm(cross(fw,F.U)),up=cross(rt,fw);
  lastView={rt,up,fw};
  // skip what can't be seen: the core and its jets when off screen or behind the Earth, everything beyond the Earth when it fills the frame
  filOn=!earthFills();coreOn=filOn&&coreVisible(rt,up,fw);
  const mono=OPT.pal===5?1:0,pal=PAL[OPT.pal];
  dayL+=((OPT.sun==="day"?1:0)-dayL)*Math.min(1,dt*2);

  // per-frame uniforms, computed in double precision and handed over camera-relative
  const BCa=new Float32Array(NB*4),BXa=new Float32Array(NB*4),BMa=new Float32Array(NB*4),BSa=new Float32Array(NB*4),BPa=new Float32Array(NB*4);
  const camSun=len(sub(cp,S.sun))/AUU;
  for(let i=0;i<NB;i++){
    const b=S.bodies[i],p=b.pos,rel=sub(p,cp),R=BODY[i].r*KM,sd=norm(sub(S.sun,p)),dsun=len(sub(S.sun,p))/AUU;
    const hide=!AST&&i!==2;
    BCa.set([rel[0],rel[1],rel[2],hide?0:R],i*4);BXa.set([...(i===2?EQ[1]:b.pole),BODY[i].f],i*4);BMa.set([...(i===2?EQ[0]:b.pm),0],i*4);
    BSa.set([...sd,clamp((camSun/dsun)**2,.03,30)],i*4);
    // brightness as a point of light: albedo, phase, distance; your eyes adapt to the local sunlight
    const D=len(rel),ca=clamp(dot(sd,mul(rel,-1/D)),-1,1),a=Math.acos(ca),phi=(Math.sin(a)+(Math.PI-a)*ca)/Math.PI;
    const flux=BODY[i].p*phi*(R/D)**2/(dsun*dsun),m=-26.74-2.5*Math.log10(Math.max(flux,1e-30))-2.5*Math.log10(Math.max(camSun*camSun,1e-6));
    BPa[i*4]=hide?0:Math.min(40,2.4*Math.pow(10,-.4*m));
  }
  const camE=sub(cp,EW),EALT=len(camE)-13;
  const sR=sunEF(true),camUp=norm(qInv(EQ,camE));
  let sD=sR;
  if(dayL>1e-3){const e=norm(len(cross([0,1,0],camUp))>1e-6?cross([0,1,0],camUp):[0,0,-1]),n=cross(camUp,e);sD=slerp(sR,norm(add(add(mul(camUp,.85),mul(e,.35)),mul(n,-.3))),dayL);}
  const mP=S.bodies[3].pos,moonEF=norm(qInv(EQ,sub(mP,EW))),moonLit=(1+dot(norm(sub(S.sun,mP)),norm(sub(EW,mP))))/2;
  const EMT=f32([EQ[0][0],EQ[1][0],EQ[2][0],EQ[0][1],EQ[1][1],EQ[2][1],EQ[0][2],EQ[1][2],EQ[2][2]]);
  const mq=[S.bodies[3].pm,cross(S.bodies[3].pole,S.bodies[3].pm),S.bodies[3].pole];
  const MMT=f32([mq[0][0],mq[1][0],mq[2][0],mq[0][1],mq[1][1],mq[2][1],mq[0][2],mq[1][2],mq[2][2]]);
  const camEF=qInv(EQ,camE),SKa=new Float32Array(16),SEa=new Float32Array(16),SNa=new Float32Array(16);
  WIN.forEach((w,k)=>{if(!w.live||!satTex)return;const K=sub(camEF,mul(w.C,13));SKa.set([K[0],K[1],K[2],w.front],k*4);SEa.set([...w.E,w.S],k*4);SNa.set([...w.N,w.texel],k*4);});
  const sunRel=sub(S.sun,cp);

  let q=null;if(TQ&&queries.length<4){q=gl.createQuery();gl.beginQuery(TQ.TIME_ELAPSED_EXT,q);}
  gl.disable(gl.BLEND);
  pass(P.scene,sceneT,u=>{
    gl.uniform2f(u("R"),sw,sh);gl.uniform1f(u("T"),t%7200);gl.uniform1f(u("DT"),diskT%7000);gl.uniform1f(u("energy"),energy);gl.uniform1f(u("voice"),voice);
    gl.uniform1f(u("surge"),surge);gl.uniform1f(u("frame"),frameN);gl.uniform1f(u("mono"),mono);gl.uniform1f(u("NETK"),OPT.net?1:0);gl.uniform1f(u("CORE"),coreOn?1:0);gl.uniform1f(u("FIL"),filOn?1:0);
    gl.uniform1f(u("focal"),1.7);gl.uniform1f(u("trailGlow"),1+energy*.4+voice*1.2+surge*.6);
    gl.uniform3fv(u("cam"),f32(cp));gl.uniform3fv(u("hot"),pal.hot);gl.uniform3fv(u("cool"),pal.cool);gl.uniform3fv(u("ACC"),hexRGB(pal.acc));
    gl.uniform3fv(u("SUNW"),f32(norm(S.sun)));
    gl.uniformMatrix3fv(u("view"),false,f32([...rt,...up,...fw]));gl.uniform1i(u("STEPS"),pr.steps);gl.uniform1i(u("ASTEPS"),pr.ast);gl.uniform1i(u("ATM"),pr.atm);
    gl.uniform3fv(u("H1"),f32([BSEP/2*Math.cos(BPH),0,BSEP/2*Math.sin(BPH)]));gl.uniform3fv(u("H2"),f32([-BSEP/2*Math.cos(BPH),0,-BSEP/2*Math.sin(BPH)]));
    gl.uniform1f(u("MH"),MH);gl.uniform1f(u("BPH"),BPH%(TWO_PI*1000));gl.uniform1f(u("BSEP"),BSEP);gl.uniform1f(u("RINGA"),RINGA%TWO_PI);
    gl.uniform4fv(u("LP"),LP);gl.uniform4fv(u("LB"),LB);gl.uniform4fv(u("LC"),LC);
    gl.uniform4fv(u("BC"),BCa);gl.uniform4fv(u("BX"),BXa);gl.uniform4fv(u("BM"),BMa);gl.uniform4fv(u("BS"),BSa);gl.uniform4fv(u("BP"),BPa);
    gl.uniform3fv(u("SUNC"),f32(sunRel));gl.uniform1f(u("SUNR"),695700*KM);
    gl.uniform3fv(u("camE"),f32(camE));gl.uniform1f(u("EALT"),EALT);gl.uniform3fv(u("ESUN"),f32(sD));gl.uniform3fv(u("EMOON"),f32(moonEF));gl.uniform1f(u("MPH"),moonLit*(1-dayL));gl.uniform1f(u("CLD"),OPT.clouds?1:0);
    gl.uniformMatrix3fv(u("EMT"),false,EMT);gl.uniformMatrix3fv(u("MMT"),false,MMT);
    gl.uniform1f(u("GTEX"),GTEX);gl.uniform1f(u("CTEX"),CTEX);
    gl.uniform4fv(u("SK"),SKa);gl.uniform4fv(u("SE"),SEa);gl.uniform4fv(u("SN"),SNa);
    gl.uniform3fv(u("NETA"),NETA);gl.uniform3fv(u("NETB"),NETB);
    tex(u,"NZ",3,noise,gl.TEXTURE_3D);tex(u,"EG",4,EGT);tex(u,"EN",5,ENT);tex(u,"EW",6,EWT);tex(u,"ECL",7,ECT);tex(u,"ETP",8,ETT);tex(u,"MOONM",10,MOONT);
    gl.activeTexture(gl.TEXTURE9);gl.bindTexture(gl.TEXTURE_2D_ARRAY,satTex||dummyArr);gl.uniform1i(u("SAT"),9);
  },sw,sh);
  pass(P.resolve,hdrT,u=>{tex(u,"S",0,sceneT.t);gl.uniform2f(u("outSize"),W,H);gl.uniform2f(u("scale"),sw/sceneT.w,sh/sceneT.h);gl.uniform2f(u("texSize"),sceneT.w,sceneT.h);gl.uniform1f(u("ca"),.01);gl.uniform1f(u("sharp"),Math.max(0,Math.min(.5,(1-sw/W)*.9)));});
  for(let i=0;i<mips.length;i++){const src=i?mips[i-1]:hdrT;pass(P.down,mips[i],u=>{tex(u,"S",0,src.t);gl.uniform2f(u("texel"),1/src.w,1/src.h);gl.uniform1i(u("first"),i?0:1);});}
  pass(P.streak,streakT,u=>{tex(u,"S",0,mips[2].t);gl.uniform2f(u("texel"),1/mips[2].w,1/mips[2].h);gl.uniform1f(u("thresh"),1.4);});
  gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE);
  for(let i=mips.length-2;i>=0;i--){const src=mips[i+1];pass(P.up,mips[i],u=>{tex(u,"S",0,src.t);gl.uniform2f(u("texel"),1/src.w,1/src.h);});}
  gl.disable(gl.BLEND);
  pass(P.comp,null,u=>{
    tex(u,"HDRI",0,hdrT.t);tex(u,"BLOOM",1,mips[0].t);tex(u,"STRK",2,streakT.t);
    gl.uniform2f(u("R"),W,H);gl.uniform1f(u("T"),t%7200);gl.uniform1f(u("bloomK"),.04+.03*energy+.04*voice+.05*surge);
    gl.uniform1f(u("streakK"),1.1);gl.uniform1f(u("exposure"),1.22);gl.uniform1f(u("cine"),OPT.cine?1:0);gl.uniform1f(u("mono"),mono);
    const h=pal.hot;gl.uniform3f(u("tint"),.55+.45*h[0],.75+.25*h[1],1.);
  });
  drawOrbits(rt,up,fw);
  if(q){gl.endQuery(TQ.TIME_ELAPSED_EXT);queries.push(q);}
  updateLabels(rt,up,fw);

  fpsFrames++;fpsT+=dt;
  if(fpsT>=.5){cv.classList.toggle("pick",pickable());if(!meEl.hidden){const k=skyAt(ME.lat,ME.lon);meEl.querySelector("em").textContent=`${phaseOf(k.sun)} · ${meFmt.format(new Date())}`;}if(openMenu==="earth")hereInfo();fps=Math.round(fpsFrames/fpsT);fpsFrames=0;fpsT=0;readout();if(openMenu==="travel")travelDistances();
    if(OPT.stats)$("stats").innerHTML=`<em>${sw}×${sh}</em> render · ${(sw/W).toFixed(2)}× → ${W}×${H}<br>${fps} fps${gpuSeen?" · GPU "+gpuMs.toFixed(1)+" ms":""} · ${pr.label} · ${HDR?"16-bit HDR":"8-bit"}${GPU?"<br>"+GPU:""}`;}
  requestAnimationFrame(frame);
}
const dummyArr=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D_ARRAY,dummyArr);gl.texImage3D(gl.TEXTURE_2D_ARRAY,0,gl.RGBA8,1,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(4));
astro(Date.now());prevQ=EQ.map(c=>c.slice());
window.jarvisBH={flyTo:(lat,lon,name)=>flyToPlace({name:name||"Place",detail:"",lat,lon,box:null,src:"Jarvis"}),go:k=>k==="hole"?goHome():k==="sun"?goSun():k==="system"?goSystem():goBody(BODY.findIndex(b=>b.key===k)),mode:m=>setMode(m)};
requestAnimationFrame(frame);
})();
