#include <gl/glut.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <string>
#define PI 3.1415926
using namespace std;
using namespace cv;


int number=0;
static GLfloat spin_x=0.0,spin_y=0.0,spin_z=0.0;
bool flag=false;
bool get_depth=false;
GLfloat cx=00.0,cy=00.0,cz=1200.0;
GLfloat near_p=500,far_p=5000.0;
ofstream outputfile,outputfile_quaternion;
float mat[16];
inline float SIGN(float x) {return (x >= 0.0f) ? +1.0f : -1.0f;}
inline float NORM(float a, float b, float c, float d) {return sqrt(a * a + b * b + c * c + d * d);}
void rotationMatrix2quaternion(float r11,float r12,float r13,float r21,float r22,float r23,float r31,float r32,float r33)
{
	float q0,q1,q2,q3;
	q0 = ( r11 + r22 + r33 + 1.0f) / 4.0f;
	q1 = ( r11 - r22 - r33 + 1.0f) / 4.0f;
	q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
	q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
	if(q0 < 0.0f) q0 = 0.0f;
	if(q1 < 0.0f) q1 = 0.0f;
	if(q2 < 0.0f) q2 = 0.0f;
	if(q3 < 0.0f) q3 = 0.0f;
	q0 = sqrt(q0);
	q1 = sqrt(q1);
	q2 = sqrt(q2);
	q3 = sqrt(q3);
	if(q0 >= q1 && q0 >= q2 && q0 >= q3) {
		q0 *= +1.0f;
		q1 *= SIGN(r32 - r23);
		q2 *= SIGN(r13 - r31);
		q3 *= SIGN(r21 - r12);
	} else if(q1 >= q0 && q1 >= q2 && q1 >= q3) {
		q0 *= SIGN(r32 - r23);
		q1 *= +1.0f;
		q2 *= SIGN(r21 + r12);
		q3 *= SIGN(r13 + r31);
	} else if(q2 >= q0 && q2 >= q1 && q2 >= q3) {
		q0 *= SIGN(r13 - r31);
		q1 *= SIGN(r21 + r12);
		q2 *= +1.0f;
		q3 *= SIGN(r32 + r23);
	} else if(q3 >= q0 && q3 >= q1 && q3 >= q2) {
		q0 *= SIGN(r21 - r12);
		q1 *= SIGN(r31 + r13);
		q2 *= SIGN(r32 + r23);
		q3 *= +1.0f;
	} else {
		printf("coding error\n");
	}
	float r;
	r = NORM(q0, q1, q2, q3);
	q0 /= r;
	q1 /= r;
	q2 /= r;
	q3 /= r;
	cout<<q0<<" "<<q1<<" "<<q2<<" "<<q3<<endl<<endl;
	outputfile_quaternion<<q0<<" "<<q1<<" "<<q2<<" "<<q3<<endl;
}

//计算相机姿态，得到结果是一个旋转矩阵3*3和相机位置3*1
void computeRT(GLfloat cx,GLfloat cy,GLfloat cz,GLfloat rx,GLfloat ry,GLfloat rz)
{
	float r11,r12,r13,r21,r22,r23,r31,r32,r33;
	/*
	r11=cos(ry*PI/180.0)*cos(rz*PI/180.0);
	r12=cos(ry*PI/180.0)*sin(rz*PI/180.0);
	r13=-sin(ry*PI/180.0);
	r21=sin(rx*PI/180.0)*sin(ry*PI/180.0)*cos(rz*PI/180.0)-cos(rx*PI/180.0)*sin(rz*PI/180.0);
	r22=sin(rx*PI/180.0)*sin(ry*PI/180.0)*sin(rz*PI/180.0)+cos(rx*PI/180.0)*cos(rz*PI/180.0);
	r23=sin(rx*PI/180.0)*cos(ry*PI/180.0);
	r31=cos(rx*PI/180.0)*sin(ry*PI/180.0)*cos(rz*PI/180.0)+sin(rx*PI/180.0)*sin(rz*PI/180.0);
	r32=cos(rx*PI/180.0)*sin(ry*PI/180.0)*sin(rz*PI/180.0)-sin(rx*PI/180.0)*cos(rz*PI/180.0);
	r33=cos(rx*PI/180.0)*cos(ry*PI/180.0);
	*/
	//改成转置
	r11=cos(ry*PI/180.0)*cos(rz*PI/180.0);
	r21=-cos(ry*PI/180.0)*sin(rz*PI/180.0);
	r31=sin(ry*PI/180.0);
	r12=sin(rx*PI/180.0)*sin(ry*PI/180.0)*cos(rz*PI/180.0)+cos(rx*PI/180.0)*sin(rz*PI/180.0);
	r22=-sin(rx*PI/180.0)*sin(ry*PI/180.0)*sin(rz*PI/180.0)+cos(rx*PI/180.0)*cos(rz*PI/180.0);
	r32=-sin(rx*PI/180.0)*cos(ry*PI/180.0);
	r13=-cos(rx*PI/180.0)*sin(ry*PI/180.0)*cos(rz*PI/180.0)+sin(rx*PI/180.0)*sin(rz*PI/180.0);
	r23=cos(rx*PI/180.0)*sin(ry*PI/180.0)*sin(rz*PI/180.0)+sin(rx*PI/180.0)*cos(rz*PI/180.0);
	r33=cos(rx*PI/180.0)*cos(ry*PI/180.0);

	/*
	r11=cos(ry*PI/180.0)*cos(rz*PI/180.0)-sin(rx*PI/180.0)*sin(ry*PI/180.0)*sin(rz*PI/180.0);
	r12=cos(ry*PI/180.0)*sin(rz*PI/180.0)+sin(rx*PI/180.0)*sin(ry*PI/180.0)*cos(rz*PI/180.0);
	r13=-cos(rx*PI/180.0)*sin(ry*PI/180.0);
	r21=-cos(rx*PI/180.0)*sin(rz*PI/180.0);
	r22=cos(rx*PI/180.0)*cos(rz*PI/180.0);
	r23=sin(rx*PI/180.0);
	r31=sin(ry*PI/180.0)*cos(rz*PI/180.0)+sin(rx*PI/180.0)*cos(ry*PI/180.0)*sin(rz*PI/180.0);
	r32=sin(ry*PI/180.0)*sin(rz*PI/180.0)-sin(rx*PI/180.0)*cos(ry*PI/180.0)*cos(rz*PI/180.0);
	r33=cos(rx*PI/180.0)*cos(ry*PI/180.0);
	*/

	float x=cx*r11+cy*r12+cz*r13;
	float y=cx*r21+cy*r22+cz*r23;
	float z=cx*r31+cy*r32+cz*r33;

// 	cout<<"x: "<<x<<endl
// 		<<"y: "<<y<<endl
// 		<<"z: "<<z<<endl;
// 	outputfile<<x<<" "<<y<<" "<<z<<" ";
//     outputfile_quaternion<<x<<" "<<y<<" "<<z<<" ";
// 	rotationMatrix2quaternion(r11,r12,r13,r21,r22,r23,r31,r32,r33);
//     outputfile<<r11<<" "<<r12<<" "<<r13<<" "
//         <<r21<<" "<<r22<<" "<<r23<<" "
//         <<r31<<" "<<r32<<" "<<r33<<endl;
    cout<<"matrix_my: "<<r11<<" "<<r12<<" "<<r13<<" "
        <<r21<<" "<<r22<<" "<<r23<<" "
        <<r31<<" "<<r32<<" "<<r33<<" "
        <<x<<" "<<y<<" "<<z<<endl;
    cout<<"matrix_gl: "<<mat[0]<<" "<<mat[4]<<" "<<mat[8]<<" "
        <<mat[1]<<" "<<mat[5]<<" "<<mat[9]<<" "
        <<mat[2]<<" "<<mat[6]<<" "<<mat[10]<<" "
        <<mat[12]<<" "<<mat[13]<<" "<<mat[14]<<endl;
}

void myDisplay(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	//相机姿态
#if 0
	gluLookAt(0.0,0.0,5.0,0.0,0.0,0.0,0.0,1.0,0.0);
#else
	gluLookAt(0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0);
	glTranslatef(cx,cy,cz);//确定相机初始位置
	glRotatef(spin_x,1.0,0.0,0.0);//相机的旋转
	glRotatef(spin_y,0.0,1.0,0.0);
	glRotatef(spin_z,0.0,0.0,1.0);
    glGetFloatv(GL_MODELVIEW_MATRIX,mat);
#endif

	//场景模型
	glColor3f(1.0,1.0,0.0);
	glPushMatrix();
		glutSolidCube(80.0);
		//glRotatef(45,0.0,1.0,0.0);
		//glTranslatef(0.0,300.0,0.0);
		//glColor3f(0.0,1.0,1.0);
		//glutSolidCube(200.0);
	glPopMatrix();
	glPushMatrix();
		glBegin(GL_POLYGON);
			glVertex3f(1600.0,-200.0,1600.0);
			glVertex3f(1600.0,-200.0,-1600.0);
			glVertex3f(-1600.0,-200.0,-1600.0);
			glVertex3f(-1600.0,-200.0,1600.0);
		glEnd();
		glBegin(GL_POLYGON);
			glVertex3f(1600.0,-200.0,1600.0);
			glVertex3f(-1600.0,-200.0,1600.0);
			glVertex3f(-1600.0,3000.0,1600.0);
			glVertex3f(1600.0,3000.0,1600.0);
		glEnd();
		glBegin(GL_POLYGON);
			glVertex3f(1600.0,-200.0,1600.0);
			glVertex3f(1600.0,-200.0,-1600.0);
			glVertex3f(1600.0,3000.0,-1600.0);
			glVertex3f(1600.0,3000.0,1600.0);
		glEnd();
		glBegin(GL_POLYGON);
			glVertex3f(-1600.0,-200.0,-1600.0);
			glVertex3f(-1600.0,-200.0,1600.0);
			glVertex3f(-1600.0,3000.0,1600.0);
			glVertex3f(-1600.0,3000.0,-1600.0);
		glEnd();
		glBegin(GL_POLYGON);
			glVertex3f(-1600.0,-200.0,-1600.0);
			glVertex3f(1600.0,-200.0,-1600.0);
			glVertex3f(1600.0,3000.0,-1600.0);
			glVertex3f(-1600.0,3000.0,-1600.0);
		glEnd();
	glPopMatrix();
	if (get_depth)
	{
		double depth_value;
		GLint viewport[4];  
		glGetIntegerv(GL_VIEWPORT, viewport);  
		int width = viewport[2];  
		int height = viewport[3];  

		int nAlignWidth = width + width%4;  
		float* pdata = new float[nAlignWidth * height];  
		memset( pdata, 0, nAlignWidth * height*sizeof(float) ); 
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glReadPixels(viewport[0], viewport[1], width, height, GL_DEPTH_COMPONENT, GL_FLOAT, pdata);  

		Mat depth = Mat::zeros( height, width, CV_16UC1 );

		for ( int i=0; i<height; i++ )
		{
			for ( int j=0; j<width; j++ )
			{
				//depth.at<float>(i,j) = pdata[(height-1-i)*width+j];
				float val = pdata[(height-1-i)*width+j];
				if (val==1)
				{
					depth.at<ushort>(i,j) = 0;
				}
				else
				{
					depth_value = (near_p*far_p/(near_p-far_p))/(val-far_p/(far_p-near_p));
					//depth_value = near_p+(far_p-near_p)*val;
					depth.at<ushort>(i,j) =(ushort)(depth_value+0.5);
				}
			}
		}


		//nrmalize( depth, depth, 0, 255, NORM_MINMAX );
		char name[4];
		itoa(number,name,10);
		string s=string(name);
		s=string(4-s.size(),'0')+s;
		cout<<s<<endl;
		++number;
		//imwrite( s+".png", depth );
		/*ofstream fp(s+".txt");
		for(int i = 0;i<height;i++)
		{
			for(int j = 0;j<width;j++)
			{
				fp<<depth.at<ushort>(i,j);
				fp<<' ';
			}
			fp<<"\t  yihang"<<'\n';
		}*/
		//depth.convertTo(depth,CV_8U);
		//imwrite( s+".png", depth );
		ofstream infp("focal_length.txt");
		double modelview[16],projection[16];
		glGetDoublev(GL_PROJECTION_MATRIX, projection);
		double fx;
		double fy;
		fx = projection[0]*(double)width/2.0;
		fy = projection[5]*(double)height/2.0;
		infp<<fx<<' '<<fy<<endl;
		delete[] pdata; 
	}
	get_depth=false;
	glutSwapBuffers();
}

void init()
{
	GLfloat mat_specular[]={1.0,1.0,1.0,0.5};
	GLfloat mat_shininess[]={50.0};
	GLfloat light_position[]={500.0,500.0,100.0,0.0};
	GLfloat white_light[]={1.0,1.0,1.0,0.5};
	GLfloat lmodel_ambient[]={0.8,0.8,0.8,0.5};
	glClearColor(0.0,0.0,0.0,0.0);
	glShadeModel(GL_SMOOTH);
	glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
	glMaterialfv(GL_FRONT,GL_SHININESS,mat_shininess);
	glLightfv(GL_LIGHT0,GL_POSITION,light_position);
	glLightfv(GL_LIGHT0,GL_DIFFUSE,white_light);
	glLightfv(GL_LIGHT0,GL_SPECULAR,white_light);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,lmodel_ambient);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);
}

void reshape(int w,int h)
{
	glViewport(0,0,(GLsizei)w,(GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(43,57.0/43,near_p,far_p);
	//glOrtho(-800,800,-600,600,near_p,far_p);
	//glFrustum(-1.0,1.0,-1.0,1.0,near_p,far_p);
	glMatrixMode(GL_MODELVIEW);
}

void keyboard(unsigned char key,int x,int y)
{
	switch(key)
	{
	case 'x':
		spin_x+=2.0;
		if (spin_x>360.0)
		{
			spin_x-=360.0;
		}
		glutPostRedisplay();
		break;
	case 'X':
		spin_x-=2.0;
		if (spin_x<0.0)
		{
			spin_x+=360.0;
		}
		glutPostRedisplay();
		break;
	case 'y':
		spin_y+=2.0;
		if (spin_y>360.0)
		{
			spin_y-=360.0;
		}
		glutPostRedisplay();
		break;
	case 'Y':
		spin_y-=2.0;
		if (spin_y<0.0)
		{
			spin_y+=360.0;
		}
		glutPostRedisplay();
		break;
	case 'z':
		spin_z+=2.0;
		if (spin_z>360.0)
		{
			spin_z-=360.0;
		}
		glutPostRedisplay();
		break;
	case 'Z':
		spin_z-=2.0;
		if (spin_z<0.0)
		{
			spin_z+=360.0;
		}
		glutPostRedisplay();
		break;
	case ' ':
		get_depth=true;
		computeRT(-cx,-cy,-cz,-spin_x,-spin_y,-spin_z);
		glutPostRedisplay();
		break;
	default:
		break;
	}
}

int main(int argc, char *argv[])
{
	outputfile.open("syntheticRT.txt");
    outputfile_quaternion.open("syntheticRT_quaternion.txt");
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutInitWindowPosition(100, 100);

	glutInitWindowSize(640, 480);

	glutCreateWindow("第一个OpenGL程序");
	init();
	glutDisplayFunc(myDisplay);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMainLoop();

	return 0;
}