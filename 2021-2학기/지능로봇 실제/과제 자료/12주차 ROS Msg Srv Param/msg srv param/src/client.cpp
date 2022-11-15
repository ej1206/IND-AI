#include <ros/ros.h>
#include <assign/msg.h>
#include <assign/srv.h>
#include <cstdlib>
#include <time.h>

assign::srv srv;
ros::ServiceClient client;
ros::Publisher pubMsg;
int iter;

void calc(const ros::TimerEvent&);


int main(int argc, char **argv)
{
    iter = 0;

    ros::init(argc, argv, "Client");

    ROS_INFO("Client");

    ros::NodeHandle n;

    pubMsg = n.advertise<assign::msg>("/assign4/msg", 1);
    client = n.serviceClient<assign::srv>("calc_area");

    ros::Timer timer = n.createTimer(ros::Duration(1), calc); // 1초에 한번씩 Service 요청

    ros::spin();

    return 0;
}

void calc(const ros::TimerEvent&){

    std::vector<int> R;      // Radius 값을 저장할 array R 생성 
    std::vector<float> Area; // Area 값을 저장할 array Area 생성

    assign::msg msg;

    srand(time(NULL));

    for(int i = 0; i< 5; i++) {
        int radius = rand()%15 + 1;; // array R에 Random Radius 값 push_back
        R.push_back(radius); 
    }

    srv.request.radius = R;         // Service Input

    if (client.call(srv))           // Service 요청 성공 시
    {
        ROS_INFO("Sub Area %d", iter);
        for(int i = 0; i< 5; i++) {
            ROS_INFO("%.2f", srv.response.area[i]);
            Area.push_back(srv.response.area[i]);  // array Area에 Service Response값 push_back
        }

        msg.header.stamp = ros::Time::now();
        msg.radius = R;       // msg의 radius를 array R로 지정
        msg.area = Area;      // msg의 area를 array Area로 지정

        pubMsg.publish(msg);  // msg publish


    }
    else
    {
        ROS_ERROR("Failed to call service add_two_ints");
    }

    iter += 1;

}
