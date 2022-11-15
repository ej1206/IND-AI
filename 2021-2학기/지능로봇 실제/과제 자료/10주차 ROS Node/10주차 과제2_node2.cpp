
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <time.h>

#include <ros/ros.h>
#include <std_msgs/Int64.h>

int main(int argc, char** argv){

    ros::init(argc, argv, "node2");                  // 2번 노드
    ros::NodeHandle nh("~");
    ros::Publisher pubB;
    pubB = nh.advertise<std_msgs::Int64>("B", 1000); // publisher 선언 -> std_msgs::Int64 type B로 publish

    ros::Rate loop_rate(10); // 10hz 주기
    srand(time(NULL));

    while (ros::ok()) {

        std_msgs::Int64 msg;
        int randNum = rand()%10 + 11;                 // 11 ~ 20 사이의 랜덤한 값

        msg.data = randNum;                           // 랜덤값을 ROS msg에 저장
        ROS_INFO("publish msg %d", msg.data);         // ROS_INFO로 msg 출력
        pubB.publish(msg);

        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}
