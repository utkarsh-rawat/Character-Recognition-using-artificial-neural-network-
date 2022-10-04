//using Eigne library
//re-check whole code


#include<math.h>
#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<Eigen>

#define l1 36		//No. of Input Neuron
#define l2 3		//No. of Hidden Neuron 
#define l3 2		//No. of Output Neuron
#define p 9			//Training Patterns

using namespace Eigen;
using namespace std;



double logsig(double z)
{
	double y;
	y=1/(1+ exp(-z));
	return y;
}



int main()
{
	MatrixXd P(l1+1,p);		//Matrix Reading all patterns 37*9 last row is bias
	MatrixXd PO(l3,p);		//Matrix Reading all patterns Targeted outputs 2*9
	VectorXd I(l1+1);
	MatrixXd W1(l2,l1+1);
	VectorXd IH(l2);
	VectorXd OH(l2+1);
	MatrixXd W2(l3,l2+1);
	VectorXd IO(l3);
	VectorXd OO(l3);		
	VectorXd TO(l3);		//Target Output,Vector	
	VectorXd Er(l3);		//Error Vector
	
	double SqEr;			//Squared Mean Error
	double eta=0.8;			//Learning Rate
	double alpha=0;			//Momentum Term
	int Itr=1;
	int MaxItr=12000;		//Maximum No. of iterations
	
	
	OH(l2+1-1)=1;			//Setting Bias of Hidden and Output layer as 1
	
	
	//Reading Pattern Matrix From a File Pattern.txt
	//Bias Value of Input and Hidden Layer is Included in Pattern.txt file
	fstream f("Pattern.txt");					//Success
	for(int i=0;i<p;i++)
	{
		double temp;
		for(int j=0;j<l1;j++)
		{
			f>>temp;
			P(j,i)=temp;
		}
		P(l1+1-1,i)=1;						//Setting bias of input and hidden layer as 1
	}
	f.close();
	
	
	//Reading Target Matrix From a File Target.txt
	fstream g("Target.txt");					//Success
	for(int i=0;i<l3;i++)
	{
		double temp;
		for(int j=0;j<p;j++)
		{
			g>>temp;
			PO(i,j)=temp;
		}
	}
	g.close();

	
	//Initializing Weight,W1
	for(int i=0;i<l2;i++)
	{
		for(int j=0;j<l1+1;j++)
		{
			W1(i,j)=(double)rand() /RAND_MAX;
		}
	}
	
	
	//Initializing Weight,W2
	for(int i=0;i<l3;i++)
	{
		for(int j=0;j<l2+1;j++)
		{
			W2(i,j)=(double)rand() /RAND_MAX;
		}
	}
	
	cout<<endl<<"Enter the Pattern for forward pass Calculation :";
	//sn=Pattern no./Serial no
	int sn;
	cin>>sn;
	
	//Selecting input pattern qand targeted output vector as per given Pattern no.
	
	I=P.col(sn-1);
	TO=PO.col(sn-1);
		
	//Forward Pass Calculation
	
	IH=W1*I;
	
	for(int i=0;i<l2;i++)
	{
		OH(i)=logsig(IH(i));
	}
	
	IO=W2*OH;
	
	for(int i=0;i<l3;i++)
	{
		OO(i)=logsig(IO(i));
	}
	
	
	//Error Calculation
	
	Er=TO-OO;
	
	SqEr=Er.norm()/sqrt(l3);	
	

	
	while(Itr<MaxItr)
	{
		
		//Decreasing learning rate every 100th iteration
		if(Itr==800)
		{
			eta=eta/2;
		}
		if(Itr==1400)
		{
			eta=eta/2;
		}

		for(sn=1;sn<p+1;sn++)
		{
			//Selecting input pattern qand targeted output vector as per given Pattern no.
			
			I=P.col(sn-1);
			TO=PO.col(sn-1);
				
			//Forward Pass Calculation
			
			IH=W1*I;
			
			for(int i=0;i<l2;i++)
			{
				OH(i)=logsig(IH(i));
			}
			
			IO=W2*OH;
			
			for(int i=0;i<l3;i++)
			{
				OO(i)=logsig(IO(i));
			}
			
			
			//Error Calculation
	
			Er=TO-OO;
			
			SqEr=Er.norm()/sqrt(l3);	
	
			
		//	cout<<endl<<"For training no. "<<ends<<Itr<<ends<<"Patter no."<<ends<<sn<<ends<<"Square Error is :"<<ends<<SqEr<<endl;
		
			if(SqEr>0.1)
			{
				//Updatiing W2(i,j)
				
				for(int i=0;i<l3;i++)
				{
					for(int j=0;j<l2;j++)
					{
						W2(i,j)=W2(i,j)+eta*(TO(i)-OO(i))*OO(i)*(1-OO(i))*OH(j)+W2(i,j)*alpha;
					}
				}
				
				//Updating W1(i,j)
				for(int i=0;i<l2;i++)
				{
					for(int j=0;j<l1;j++)
					{
						double sum=0;
						for(int k=0;k<l3;k++)
						{
							sum=sum+(TO(k)-OO(k))*OO(k)*(1-OO(k))*W2(k,i)*OH(i)*(1-OH(i))*I(j);
						}
						W1(i,j)=W1(i,j)+eta*sum+W1(i,j)*alpha;
					}
				}
			}
		}
	Itr++;
	}
	
	
	
	
	//Test for a particular pattern
	TEST:cout<<endl<<"Enter the Pattern for Test case :";
	//sn=Pattern no./Serial no
	cin>>sn;
	
	//Selecting input pattern qand targeted output vector as per given Pattern no.
	
	I=P.col(sn-1);
	TO=PO.col(sn-1);
		
	//Forward Pass Calculation
	
	IH=W1*I;
	
	for(int i=0;i<l2;i++)
	{
		OH(i)=logsig(IH(i));
	}
	
	IO=W2*OH;
	
	for(int i=0;i<l3;i++)
	{
		OO(i)=logsig(IO(i));
	}
	
	
	//Error Calculation
	
	Er=TO-OO;
	
	SqEr=Er.norm()/sqrt(l3);	
	
	
	ofstream h("W2.txt");
	h<<"W2 Weight Matrix"<<endl;
	h<<W2;
	h.close();

	ofstream i("W1.txt");
	i<<"W1 Weight Matrix"<<endl;
	i<<W2;
	i.close();
	
	
	cout<<"Total Error is :"<<ends<<SqEr<<endl<<"Targeted Values"<<endl<<TO<<endl<<"Obtained Output"<<endl<<OO<<endl;

	if(SqEr<0.1)
	{
		cout<<"Recognised"<<endl;
	}
	else
	{
		cout<<"Not Recognised"<<endl;
	}
	
	if(sn<10)
	{
		goto TEST;
	}
}
	
	
	
