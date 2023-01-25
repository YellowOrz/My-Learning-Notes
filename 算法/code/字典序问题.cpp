#include <iostream>
#include <fstream>

using namespace std;
// 求阶乘
int factorial(int n){
    int fc = 1;
    for(int i=1;i<=n;i++) fc*=i;
    return fc;
}
// 组合数公式，m里面选n个
int combination (int m, int n){
    if (n>m){
        cout << "Error: input wrong number!" << endl;
        exit(0);
    }
    int choice = 1;
    for(int i = m;i>m-n;i--)
        choice *= i;
    return choice/factorial(n);
}
int code(string letters){// l表示可用字母个数
    int length = letters.length(),location = 0;

    // 先计算长度小于length的所有情况，也就是找到长度为length的起始位置
    for(int i=1;i<length;i++)
        location+=combination(26,i);

    // 计算从头到尾每一个字母的起始位置
    int temp = 'a'-'a';
    for(int j = 1;j<=length;j++){
        for(int i = temp; i < letters[j-1]-'a'; i++)
            location+=combination(25-i,length-j);   //对于第c个字符©，它的前面有 从它前一位的字母字符®到©打头的字符串，长度分别为C_{25-i}^{length-c},i=(®+1-'a')...(©-'a')
        temp=letters[j-1]+1-'a';
    }
    return location+1;
}
int main(){
    ifstream fp("../input.txt");
    int k = 0;
    fp>>k;
    for(int i = 0;i<k;i++){
        string letter;
        fp>>letter;
        //判断是否有非法字符
        for(char& a:letter)
            if (a<'a' || a>'z'){
                cout << "Error: there has non-letter string!" << endl;
                exit(-1);
            }

        cout << code(letter) << endl;
    }
    return 0;
}