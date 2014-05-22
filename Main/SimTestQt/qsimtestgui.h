#ifndef QSIMTESTGUI_H
#define QSIMTESTGUI_H

#include "KlamptQt/qtguibase.h"
#include "KlamptQt/qklamptdisplay.h"
#include "qsimtestbackend.h"

#include <Interface/GenericGUI.h>
#include "Modeling/World.h"
#include <Interface/SimTestGUI.h>

#include <QMouseEvent>
#include <QObject>
#include <QTimer>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>

#include <driveredit.h>
#include <logoptions.h>
#include <controllerdialog.h>

class QSimTestGUI : public QtGUIBase
{
    Q_OBJECT

public:
    QSimTestGUI(QKlamptDisplay* display,SimTestBackend* _backend);
    virtual ~QSimTestGUI();
    QKlamptDisplay* display;
    WorldSimulation* sim;
    DriverEdit *driver_tool;
    LogOptions *log_options;
    ControllerDialog *controller_dialog;
    QString old_filename;
    QSettings* ini;

    int constrain_mode,constrain_point_mode,delete_mode;
    virtual bool OnCommand(const string& cmd,const string& args);
    virtual bool OnRefresh();
    void UpdateGUI();
    void UpdateMeasurements();
    void LoadFile(QString filename=QString());
    void LoadFilePrompt(QString directory_key=".",QString filter="*");
    void SaveScenario(QString filename=QString());
    void SaveLastScenario();
    void ShowHelp();
    void ShowAbout();
public slots:
    void SendMousePress(QMouseEvent *e);
    void SendDriverValue(int index, float value);
    void ShowSensor(int sensor);
    void HideSensor(int sensor);
    void SendMeasurement(int sensor, int measurement, bool status);
    void SendControllerSetting(int robot,string setting, string value);
    void SendControllerCommand(int robot,string setting, string value);
    void SendConnection(int robot,QString host,int port,int rate);
signals:
    void  EndDelete();
    void EndConstrain();
};

#endif // QTGUIBASE_H
