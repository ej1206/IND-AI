
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <time.h>

#include <ros/ros.h>
#include <std_msgs/Int64.h>

void subscribeA(const std_msgs::Int64 msg);
void subscribeB(const std_msgs::Int64 msg);
void publishC(const ros::TimerEvent&);

int64_t A, B, C;                                          // 전역변수 A, B, C
ros::Publisher pub_c;

int main(int argc, char** argv){

    ros::init(argc, argv, "node3");                       // 3번 노드
    ros::NodeHandle nh("~");
    ros::Subscriber sub_A, sub_B;

    sub_A = nh.subscribe("/node1/A", 1000, subscribeA);   // "/node1/A" topic subscribe
    sub_B = nh.subscribe("/node2/B", 1000, subscribeB);   // "/node2/B" topic subscribe

    pub_c = nh.advertise<std_msgs::Int64>("C", 1000);     // publisher 지정 -> std_msgs::Int64 type C로 publish

    ros::Timer timer = nh.createTimer(ros::Duration(0.1), publishC); // timer 설정하여, 0.1주기로 publishC 함수 호출

    ros::spin();

    return 0;
}

void subscribeA(const std_msgs::Int64 msg) {
    A = msg.data;                                         // 전역변수 A에 msg topic 값 저장

}
void subscribeB(const std_msgs::Int64 msg) {
    B = msg.data;                                         // 전역변수 B에 msg topic 값 저장
}

void publishC(const ros::TimerEvent&) {

    C = A + B;                                            // A + B 연산
    ROS_INFO("A, B, C = %d, %d, %d", A, B, C);            // ROS_INFO로 A, B, C출력
    std_msgs::Int64 msg;
    msg.data = C;                                         // C 값을 ROS msg에 저장
    pub_c.publish(msg);

}
