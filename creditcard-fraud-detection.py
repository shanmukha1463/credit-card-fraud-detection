import numpy as np
X=np.loadtxt('creditcard.csv',dtype=float,delimiter=',',skiprows=1,usecols=(range(0,30)));



Y=np.loadtxt('creditcard.csv',dtype=str,delimiter=',',skiprows=1,usecols=30)

(rows,columns)=X.shape

y=np.arange(0,rows)

bigx=0;
for i in range(rows):
    if(Y[i][1]=="0"):
        y[i]=0
        bigx+=1
    else:
        y[i]=1

rng_state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(rng_state)
np.random.shuffle(y)
x_train_rows=int(0.6*bigx)
x_train=np.zeros((x_train_rows,columns));
x_test=np.zeros((rows-x_train_rows,columns));
y_test=np.zeros((rows-x_train_rows,1));
con=0;pan=0;
for i in range(rows):
    if(y[i]==0 and con<x_train_rows):
        x_train[con]=(X[i]);
        con+=1;
    else:
        x_test[pan]=X[i];
        y_test[pan]=y[i]
        pan+=1;
        
        
    
        
        

meanf=np.zeros((columns,1))
maximum=np.zeros(columns)
minimum=np.zeros(columns)
for i in range(columns):
    su=0
    mi=float('inf')
    ma=float('-inf')
    for j in range(x_train_rows):
        su=su+x_train[j][i]
        if(mi>x_train[j][i]):
            mi=x_train[j][i]
        if(ma<x_train[j][i]):
            ma=x_train[j][i]
    meanf[i][0]=(su/x_train_rows)
    maximum[i]=ma
    minimum[i]=mi
    
        

for i in range(columns):
    for j in range(x_train_rows):
        x_train[j][i]=(x_train[j][i]-meanf[i])/(maximum[i]-minimum[i])

epsilon=[0.000000001,0.00001,0.001, 0.003, 0.007 ,0.01 ,0.03, 0.07, 0.1, 0.3, 0.7]
#epsilon=[0.001]

mean=np.zeros((columns,1))
for i in range(columns):
    su=0
    for j in range(x_train_rows):
        su=su+x_train[j][i]
    mean[i][0]=(su/x_train_rows)
x_train_matrix=np.matrix(x_train)
mean_matrix=np.matrix(mean)
hh=np.zeros((x_train_rows,columns))
for i in range(x_train_rows):
    hh[i]=mean.T
hh_matrix=np.matrix(hh)
covar_matrix=(1/x_train_rows)*(((x_train_matrix-hh).T)*(x_train_matrix-hh));
val=((np.linalg.det(covar_matrix)**(1/2))*((2*np.pi)**(columns/2)))
    

(x_test_rows,x_test_columns)=x_test.shape
for i in range(columns):
    for j in range(x_test_rows):
        x_test[j][i]=(x_test[j][i]-meanf[i])/(maximum[i]-minimum[i])
best_f1=-1;
for e in epsilon:
    y_cal=np.zeros((x_test_rows,1))
    for i in range(x_test_rows):
        a=x_test[i]
        a=(np.matrix(a)).T
        p=(1/val)*(np.exp(-(1/2)*((a-mean_matrix).T)*(covar_matrix.I)*(a-mean_matrix)))
        if(p<e):
            y_cal[i]=1
        else:
            y_cal[i]=0
    tp=float(0)
    fp=float(0)
    fn=float(0)
    for i in range(x_test_rows):
        if(y_test[i]==1 and y_cal[i]==1):
            tp+=1
        elif(y_test[i]==0 and y_cal[i]==1):
            fp+=1
        elif(y_test[i]==1 and y_cal[i]==0):
            fn+=1
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=(2*precision* recall)/(precision+recall);
    if(best_f1<f1):
        best_f1=f1
        best_e=e
print(best_f1)
print(best_e)
        