#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
void RepeatNum(vector<int> &S,int &mode,int &count){
    vector<int> statistic;
    for(int s:S){
        if(statistic.size()<(s+1)) statistic.resize(s+1);
        statistic[s]++;
    }
    count=-1,mode=-1;
    for(int i = 1;i<statistic.size();i++)
        if(count<statistic[i]){
            count = statistic[i];
            mode = i;
        }
}

int main() {
    ifstream fp("../input.txt");
    int n = 0; fp>>n;
    vector<int> S;
    while(n!=0){
        int temp; fp>>temp;
        S.push_back(temp);
        n--;
    }
    int mode=0,count=0;
    RepeatNum(S,mode,count);
    cout <<mode<<" "<<count<<endl;
    return 0;
}