function plot_PR_F1()
    % ����ȫ��Ĭ������ΪTimes New Roman
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultTextFontName', 'Times New Roman');
    
    plot_PR(); 
    plot_F1(); 
end

function plot_PR()
    pr_csv_dict = struct(...
        'YOLOv5n', 'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\yolov5n5\PR_curve.csv', ...
        'FasterYOLO', 'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\FasterYOLO\PR_curve.csv' ...
    );

    fig = figure('Name','P-R Curve','Position',[100 100 800 600]);
    ax = axes('Parent',fig);
    
    % ��ʽ�������������壨�����Լӹ̣�
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 24); 
    hold(ax,'on');
    
    model_names = fieldnames(pr_csv_dict);
    for i = 1:length(model_names)
        name = model_names{i};
        path = pr_csv_dict.(name);
        data = csvread(path,1,0);
        plot(ax, data(:,2), data(:,3), 'LineWidth',2, 'DisplayName',name);
    end
    
    % ���ñ�ǩ����
    xlabel(ax, 'Recall', 'FontName', 'Times New Roman', 'FontSize', 24);
    ylabel(ax, 'Precision', 'FontName', 'Times New Roman', 'FontSize', 24);
    
    % ����ͼ������
    h_legend = legend(ax, 'Location','NorthEastOutside');
    set(h_legend, 'FontName', 'Times New Roman', 'FontSize', 24);
    
    % ���귶Χ������
    xlim(ax, [0 1]);
    ylim(ax, [0 1]);
    grid(ax, 'off');
    
    % ����ͼ��ȷ������Ƕ�룩
    print(fig, 'pr_curve', '-dpng', '-r250 -cmyk');
    hold(ax,'off');
end

function plot_F1()
    f1_csv_dict = struct(...
        'YOLOv5n', 'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\yolov5n5\F1_curve.csv', ...
        'FasterYOLO', 'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\FasterYOLO\F1_curve.csv' ...
    );

    fig = figure('Name','F1 Curve','Position',[100 100 800 600]);
    ax = axes('Parent',fig);
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 12);
    hold(ax,'on');
    
    model_names = fieldnames(f1_csv_dict);
    for i = 1:length(model_names)
        name = model_names{i};
        path = f1_csv_dict.(name);
        data = csvread(path,1,0);
        plot(ax, data(:,2), data(:,6), 'LineWidth',2, 'DisplayName',name);
    end
    
    xlabel(ax, 'Confidence', 'FontName', 'Times New Roman', 'FontSize', 14);
    ylabel(ax, 'F1 Score', 'FontName', 'Times New Roman', 'FontSize', 14);
    
    h_legend = legend(ax, 'Location','NorthEastOutside');
    set(h_legend, 'FontName', 'Times New Roman', 'FontSize', 12);
    
    xlim(ax, [0 1]);
    ylim(ax, [0 1]);
    grid(ax, 'on');
    
    print(fig, 'f1_curve', '-dpng', '-r250 -cmyk');
    hold(ax,'off');
end