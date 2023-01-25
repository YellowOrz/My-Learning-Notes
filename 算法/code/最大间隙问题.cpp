#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;
float find_min(const vector<float> &nums){
    float min=nums[0];
    for(float n:nums)
        min = min<n?min:n;
    return min;
}
float find_max(const vector<float> &nums){
    float max=nums[0];
    for(float n:nums)
        max = max>n?max:n;
    return max;
}
float max_gap(const vector<float> &nums){
    // 计算区间大小
    float max=find_max(nums),min=find_min(nums);
    float len = (max-min)/(nums.size()-1);

    // 初始化区间，区间个数为数字个数-1
    vector<vector<float>> nums_order;
    nums_order.resize(nums.size());

    // 把数字放入区间
    for(float n:nums){
        int order = (n-min)/len;
        nums_order[order].push_back(n);
    }

    // 找到空区间
    for(int i=0;i<nums_order.size();i++)
        if(nums_order[i].empty())
            return find_min(nums_order[i+1]) -find_max(nums_order[i-1]);

    return -1.0;
}
int main(){
    ifstream fp("../input.txt");
    int k = 0;
    vector<float> nums;
    fp>>k;
    while(k--){
        float num;
        fp>>num;
        nums.push_back(num);
    }
    cout << max_gap(nums) <<endl;
    return 0;
}