#include <ros/ros.h>
#include <std_msgs/Int64.h>

int sum, num;                                          // 전역변수 sum, num 선언
std::vector<int> array;
void subscribeC(const std_msgs::Int64 msg);

int main(int argc, char** argv){

    ros::init(argc, argv, "node4");                    // 4번 노드
    ros::NodeHandle nh("~");
    ros::Subscriber sub_C;
    sub_C= nh.subscribe("/node3/C", 1000, subscribeC); // "/node3/C" topic subscribe

    sum, num = 0;                                      // 전역변수 sum, num 초기화

    ros::spin();

    return 0;
}

void subscribeC(const std_msgs::Int64 msg) {

    int C = msg.data;
    sum += C;
    num += 1;
    double cumulativeAvg = double(sum) / double(num);                                // C의 누적평균 계산

    ROS_INFO("C : %d, sum : %d, Cumulative Average : %.2lf", C, sum, cumulativeAvg); // C의 누적평균 출력

}
