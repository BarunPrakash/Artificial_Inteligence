# include<iostream>
#include<stdio.h>
//#include<stdboll.h>
using namespace std;

#define n 4
#define m 4

class AritificialIntelligenceUsingChass
{
	
	
	private:
	int MazematSolMtrix[n][m];
	public:
	bool solveChessUntilUsingAritificallearning(int Mazemat[n][n],int rightMov ,int downMov ,int pastlarningsol[n][n]);
	inline void printChessUtil(int Mazemat[n][n])
	{
		for(int irow=0;irow<n;irow++)
		{
			for(int icol=0;icol<m;icol++)
			{
				printf("%3d",Mazemat[irow][icol]);
			}
			printf("\n");
		}
	}
	inline void fillWithZeroChessUtil(int Mazemat[n][n])
        {
                for(int irow=0;irow<n;irow++)
                {
                        for(int icol=0;icol<m;icol++)
                        {   
                        	Mazemat[irow][icol]=0;       
                        }
                        
                }
        }
	bool startLearnig(int solMt[n][n])
	{
		//memset(MazematSolMtrix ,0,sizeof(MazematSolMtrix[0][0])*n*n);
		fillWithZeroChessUtil(solMt);
		if(solveChessUntilUsingAritificallearning(solMt,0,0,MazematSolMtrix)==false)
		{
			cout<<"Son doesnot exit!!"<<'\n';
			return false;
		}
		printChessUtil(MazematSolMtrix);
		return true;
	}

	bool isSafeMazemat(int Mazemat[n][n],int iRow ,int iCol)
	{
		if(iRow>=0 && iRow<n && iCol>=0 && iCol<n && Mazemat[iRow][iCol]==1)
			return true;
		false;
	}
	


};


bool AritificialIntelligenceUsingChass ::solveChessUntilUsingAritificallearning(int Mazemat[n][n],int rightMov ,\
int downMov ,int pastlarningsol[n][n])

{
	if(rightMov==n-1 &&downMov ==n-1)
	{
		pastlarningsol[rightMov][downMov]=1;
		return true;
	}

	if(isSafeMazemat(Mazemat ,rightMov,downMov)==true)
	{
		pastlarningsol[rightMov][downMov]=1;
		

		if(solveChessUntilUsingAritificallearning(Mazemat ,rightMov+1,downMov ,pastlarningsol)==true)
			return true;
		
		if(solveChessUntilUsingAritificallearning(Mazemat ,rightMov,downMov+1 ,pastlarningsol)==true)
                        return true;

		if(pastlarningsol[rightMov][downMov]==false)
			return false;

	}
	return false;


}
