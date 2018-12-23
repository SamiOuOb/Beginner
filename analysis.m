clc
clear all
close all

%%% Read all records of UE1 ~ UE4
%%% UE1: A1; UE2: Redmi; UE3: Zenfone; UE4: Xiaomi3
wifi_record_UE1 = 'wifi_record_UE1.txt';
wifi_record_UE2 = 'wifi_record_UE2.txt';
wifi_record_UE3 = 'wifi_record_UE3.txt';
wifi_record_UE4 = 'wifi_record_UE4.txt';

% fid_wifi_record_UE1 = fopen(wifi_record_UE1,'r');
% fid_wifi_record_UE2 = fopen(wifi_record_UE2,'r');
% fid_wifi_record_UE3 = fopen(wifi_record_UE3,'r');
% fid_wifi_record_UE4 = fopen(wifi_record_UE4,'r');

%%% split wifi records
UE1 = splitRecords(wifi_record_UE1);
nUE1 = noRepeatBSSID(UE1);
newUE1 = modifyTable(nUE1);
dividedUE1_1 = timeDividedTable(nUE1,1);
dividedUE1_2 = timeDividedTable(nUE1,2);


% sameind = find(ismember(A{:,1},A{ind(index),1}));
UE2 = splitRecords(wifi_record_UE2);
nUE2 = noRepeatBSSID(UE2);
newUE2 = modifyTable(nUE2);
dividedUE2_1 = timeDividedTable(nUE2,1);
dividedUE2_2 = timeDividedTable(nUE2,2);

UE3 = splitRecords(wifi_record_UE3);
nUE3 = noRepeatBSSID(UE3);
newUE3 = modifyTable(nUE3);
dividedUE3_1 = timeDividedTable(nUE3,1);
dividedUE3_2 = timeDividedTable(nUE3,2);

UE4 = splitRecords(wifi_record_UE4);
nUE4 = noRepeatBSSID(UE4);
newUE4 = modifyTable(nUE4);
dividedUE4_1 = timeDividedTable(nUE4,1);
dividedUE4_2 = timeDividedTable(nUE4,2);


%%% read BSSID and AP-scanned times
UE1_BSSID_1 = table2cell(newUE1(:,1));
UE1_scan_count_1 = table2array(newUE1(:,2));

UE2_BSSID_1 = table2cell(newUE2(:,1));
UE2_scan_count_1 = table2array(newUE2(:,2));

UE3_BSSID_1 = table2cell(newUE3(:,1));
UE3_scan_count_1 = table2array(newUE3(:,2));

UE4_BSSID_1 = table2cell(newUE4(:,1));
UE4_scan_count_1 = table2array(newUE4(:,2));
% 

%%% calculate Jaccard index for each pair of UEs

% UE1 and UE2 are carried by the same user and both on the same book
Jaccard_index_UE1_UE2 = Jaccard_cal(UE1_BSSID_1,UE2_BSSID_1); 

% UE3 and UE4 are carried by the same user and one is on the book, the
% other is in the backpack
Jaccard_index_UE3_UE4 = Jaccard_cal(UE3_BSSID_1,UE4_BSSID_1); 

% UE1 and UE3 are carried by different users
Jaccard_index_UE1_UE3 = Jaccard_cal(UE1_BSSID_1,UE3_BSSID_1); 
Jaccard_index_UE2_UE3 = Jaccard_cal(UE2_BSSID_1,UE3_BSSID_1); 
% 
% Jac_txt = fopen('Jaccard_index.txt','a');
% fprintf(Jac_txt,'%f ', Jaccard_index);

%%% combine as a table
UE1_UE2_table = combine(UE1_BSSID_1, UE1_scan_count_1,UE2_BSSID_1, UE2_scan_count_1);
UE3_UE4_table = combine(UE3_BSSID_1, UE3_scan_count_1,UE4_BSSID_1, UE4_scan_count_1);
UE1_UE3_table = combine(UE1_BSSID_1, UE1_scan_count_1,UE3_BSSID_1, UE3_scan_count_1);
UE2_UE3_table = combine(UE2_BSSID_1, UE2_scan_count_1,UE3_BSSID_1, UE3_scan_count_1);

UE1_UE2_weightTable = combineWeightedTable(newUE1{:,1},newUE1{:,3},newUE2{:,1},newUE2{:,3});
UE1_UE2_weightTable{:,1} = UE1_UE2_weightTable{:,1} / sum(UE1_UE2_weightTable{:,1});
UE1_UE2_weightTable{:,2} = UE1_UE2_weightTable{:,2} / sum( UE1_UE2_weightTable{:,2});
% UE1_UE2_weightTable{:,1} = UE1_UE2_weightTable{:,1} / sum(UE1_UE2_table{:,1});
% UE1_UE2_weightTable{:,2} = UE1_UE2_weightTable{:,2} / sum(UE1_UE2_table{:,3});

UE3_UE4_weightTable = combineWeightedTable(newUE3{:,1},newUE3{:,3},newUE4{:,1},newUE4{:,3});
UE3_UE4_weightTable{:,1} = UE3_UE4_weightTable{:,1} / sum( UE3_UE4_weightTable{:,1});
UE3_UE4_weightTable{:,2} = UE3_UE4_weightTable{:,2} / sum(UE3_UE4_weightTable{:,2});
% UE3_UE4_weightTable{:,1} = UE3_UE4_weightTable{:,1} / sum(UE3_UE4_table{:,1});
% UE3_UE4_weightTable{:,2} = UE3_UE4_weightTable{:,2} / sum(UE3_UE4_table{:,3});

UE1_UE3_weightTable = combineWeightedTable(newUE1{:,1},newUE1{:,3},newUE3{:,1},newUE3{:,3});
UE1_UE3_weightTable{:,1} = UE1_UE3_weightTable{:,1} / sum(UE1_UE3_weightTable{:,1});
UE1_UE3_weightTable{:,2} = UE1_UE3_weightTable{:,2} / sum(UE1_UE3_weightTable{:,2});
% UE1_UE3_weightTable{:,1} = UE1_UE3_weightTable{:,1} / sum(UE1_UE3_table{:,1});
% UE1_UE3_weightTable{:,2} = UE1_UE3_weightTable{:,2} / sum(UE1_UE3_table{:,3});

UE2_UE3_weightTable = combineWeightedTable(newUE2{:,1},newUE2{:,3},newUE3{:,1},newUE3{:,3});
UE2_UE3_weightTable{:,1} = UE2_UE3_weightTable{:,1} / sum(UE2_UE3_weightTable{:,1});
UE2_UE3_weightTable{:,2} = UE2_UE3_weightTable{:,2} / sum(UE2_UE3_weightTable{:,2});
% UE2_UE3_weightTable{:,1} = UE2_UE3_weightTable{:,1} / sum(UE2_UE3_table{:,1});
% UE2_UE3_weightTable{:,2} = UE2_UE3_weightTable{:,2} / sum(UE2_UE3_table{:,3});

%%% sub-interval
UE1_UE2_weightTable2_1 = combineWeightedTable(dividedUE1_1{:,1},dividedUE1_1{:,3},dividedUE2_1{:,1},dividedUE2_1{:,3});
UE1_UE2_weightTable2_1{:,1} = UE1_UE2_weightTable2_1{:,1} / sum(UE1_UE2_weightTable2_1{:,1});
UE1_UE2_weightTable2_1{:,2} = UE1_UE2_weightTable2_1{:,2} / sum(UE1_UE2_weightTable2_1{:,2});
% UE1_UE2_weightTable2_1{:,1} = UE1_UE2_weightTable2_1{:,1} / sum(dividedUE1_1{:,2});
% UE1_UE2_weightTable2_1{:,2} = UE1_UE2_weightTable2_1{:,2} / sum(dividedUE2_1{:,2});

UE1_UE2_weightTable2_2 = combineWeightedTable(dividedUE1_2{:,1},dividedUE1_2{:,3},dividedUE2_2{:,1},dividedUE2_2{:,3});
UE1_UE2_weightTable2_2{:,1} = UE1_UE2_weightTable2_2{:,1} / sum(UE1_UE2_weightTable2_2{:,1});
UE1_UE2_weightTable2_2{:,2} = UE1_UE2_weightTable2_2{:,2} / sum(UE1_UE2_weightTable2_2{:,2});
% UE1_UE2_weightTable2_2{:,1} = UE1_UE2_weightTable2_2{:,1} / sum(dividedUE1_2{:,2});
% UE1_UE2_weightTable2_2{:,2} = UE1_UE2_weightTable2_2{:,2} / sum(dividedUE2_2{:,2});

UE3_UE4_weightTable2_1 = combineWeightedTable(dividedUE3_1{:,1},dividedUE3_1{:,3},dividedUE4_1{:,1},dividedUE4_1{:,3});
UE3_UE4_weightTable2_1{:,1} = UE3_UE4_weightTable2_1{:,1} / sum(UE3_UE4_weightTable2_1{:,1});
UE3_UE4_weightTable2_1{:,2} = UE3_UE4_weightTable2_1{:,2} / sum(UE3_UE4_weightTable2_1{:,2});
% UE3_UE4_weightTable2_1{:,1} = UE3_UE4_weightTable2_1{:,1} / sum(dividedUE3_1{:,2});
% UE3_UE4_weightTable2_1{:,2} = UE3_UE4_weightTable2_1{:,2} / sum(dividedUE4_1{:,2});

UE3_UE4_weightTable2_2 = combineWeightedTable(dividedUE3_2{:,1},dividedUE3_2{:,3},dividedUE4_2{:,1},dividedUE4_2{:,3});
UE3_UE4_weightTable2_2{:,1} = UE3_UE4_weightTable2_2{:,1} / sum(UE3_UE4_weightTable2_2{:,1});
UE3_UE4_weightTable2_2{:,2} = UE3_UE4_weightTable2_2{:,2} / sum(UE3_UE4_weightTable2_2{:,2});
% UE3_UE4_weightTable2_2{:,1} = UE3_UE4_weightTable2_2{:,1} / sum(dividedUE3_2{:,2});
% UE3_UE4_weightTable2_2{:,2} = UE3_UE4_weightTable2_2{:,2} / sum(dividedUE4_2{:,2});


UE1_UE3_weightTable2_1 = combineWeightedTable(dividedUE1_1{:,1},dividedUE1_1{:,3},dividedUE3_1{:,1},dividedUE3_1{:,3});
UE1_UE3_weightTable2_1{:,1} = UE1_UE3_weightTable2_1{:,1} / sum(UE1_UE3_weightTable2_1{:,1});
UE1_UE3_weightTable2_1{:,2} = UE1_UE3_weightTable2_1{:,2} / sum(UE1_UE3_weightTable2_1{:,2});
% UE1_UE3_weightTable2_1{:,1} = UE1_UE3_weightTable2_1{:,1} / sum(dividedUE1_1{:,2});
% UE1_UE3_weightTable2_1{:,2} = UE1_UE3_weightTable2_1{:,2} / sum(dividedUE3_1{:,2});

UE1_UE3_weightTable2_2 = combineWeightedTable(dividedUE1_2{:,1},dividedUE1_2{:,3},dividedUE3_2{:,1},dividedUE3_2{:,3});
UE1_UE3_weightTable2_2{:,1} = UE1_UE3_weightTable2_2{:,1} / sum(UE1_UE3_weightTable2_2{:,1});
UE1_UE3_weightTable2_2{:,2} = UE1_UE3_weightTable2_2{:,2} / sum(UE1_UE3_weightTable2_2{:,2});
% UE1_UE3_weightTable2_2{:,1} = UE1_UE3_weightTable2_2{:,1} / sum(dividedUE1_2{:,2});
% UE1_UE3_weightTable2_2{:,2} = UE1_UE3_weightTable2_2{:,2} / sum(dividedUE3_2{:,2});

UE2_UE3_weightTable2_1 = combineWeightedTable(dividedUE2_1{:,1},dividedUE2_1{:,3},dividedUE3_1{:,1},dividedUE3_1{:,3});
UE2_UE3_weightTable2_1{:,1} = UE2_UE3_weightTable2_1{:,1} / sum(UE2_UE3_weightTable2_1{:,1});
UE2_UE3_weightTable2_1{:,2} = UE2_UE3_weightTable2_1{:,2} / sum(UE2_UE3_weightTable2_1{:,2});
% UE2_UE3_weightTable2_1{:,1} = UE2_UE3_weightTable2_1{:,1} / sum(dividedUE2_1{:,2});
% UE2_UE3_weightTable2_1{:,2} = UE2_UE3_weightTable2_1{:,2} / sum(dividedUE3_1{:,2});


UE2_UE3_weightTable2_2 = combineWeightedTable(dividedUE2_2{:,1},dividedUE2_2{:,3},dividedUE3_2{:,1},dividedUE3_2{:,3});
UE2_UE3_weightTable2_2{:,1} = UE2_UE3_weightTable2_2{:,1} / sum(UE2_UE3_weightTable2_2{:,1});
UE2_UE3_weightTable2_2{:,2} = UE2_UE3_weightTable2_2{:,2} / sum(UE2_UE3_weightTable2_2{:,2});
% UE2_UE3_weightTable2_2{:,1} = UE2_UE3_weightTable2_2{:,1} / sum(dividedUE2_2{:,2});
% UE2_UE3_weightTable2_2{:,2} = UE2_UE3_weightTable2_2{:,2} / sum(dividedUE3_2{:,2});


UE1_UE2_TOTAL = join(UE1_UE2_table,UE1_UE2_weightTable,'Keys','RowNames');
UE3_UE4_TOTAL = join(UE3_UE4_table,UE3_UE4_weightTable,'Keys','RowNames');
UE1_UE3_TOTAL = join(UE1_UE3_table,UE1_UE3_weightTable,'Keys','RowNames');
UE2_UE3_TOTAL = join(UE2_UE3_table,UE2_UE3_weightTable,'Keys','RowNames');
% T = join(UE1_UE2_TOTAL,UE3_UE4_TOTAL);

%%%%%%%%%%%%%%%%%%%%% calculate similarity %%%%%%%%%%%%%%%%%
% MacroCoef = 0:0.25:1.0;

% all records
% originA = UE1_UE2_TOTAL{:,1};
% originB = UE1_UE2_TOTAL{:,3};

%%%%%%%%%%%% similarity of UE1 & UE2 %%%%%%%%%%
UE1_relative_frequency = UE1_UE2_TOTAL{:,2};
UE2_relative_frequency = UE1_UE2_TOTAL{:,4};
% rAB = [UE1_relative_frequency,UE2_relative_frequency];
UE1_weighted = UE1_UE2_TOTAL{:,5};
UE2_weighted = UE1_UE2_TOTAL{:,6};
UE1_UE2_weighted = [UE1_weighted,UE2_weighted];
UE1_weighted_2_1 = UE1_UE2_weightTable2_1{:,1};
UE2_weighted_2_1 = UE1_UE2_weightTable2_1{:,2};
UE1_UE2_weighted_2_1 = [UE1_weighted_2_1,UE2_weighted_2_1];
UE1_weighted_2_2 = UE1_UE2_weightTable2_2{:,1};
UE2_weighted_2_2 = UE1_UE2_weightTable2_2{:,2};
UE1_UE2_weighted_2_2 = [UE1_weighted_2_2,UE2_weighted_2_2];

% Euclidean distance
UE1_UE2_Euclidean = pdist(UE1_UE2_weighted','euclidean');

% Cos = 1 - pdist(rAB','cosine')
UE1_UE2_Cos_w = 1 - pdist(UE1_UE2_weighted','cosine');
UE1_UE2_Cos_w2_1 = 1 - pdist(UE1_UE2_weighted_2_1','cosine');
UE1_UE2_Cos_w2_2 = 1 - pdist(UE1_UE2_weighted_2_2','cosine');

UE1_UE2_Similarity = (UE1_UE2_Cos_w2_1 + UE1_UE2_Cos_w2_2 ) / 2;
%%%%%%%%%%%%%%%%%%% similarity of UE3 & UE4 %%%%%%%%%%%%%%%%%%%%%

UE3_relative_frequency = UE3_UE4_TOTAL{:,2};
UE4_relative_frequency = UE3_UE4_TOTAL{:,4};
% rAB = [UE3_relative_frequency,UE4_relative_frequency];
UE3_weighted = UE3_UE4_TOTAL{:,5};
UE4_weighted = UE3_UE4_TOTAL{:,6};
UE3_UE4_weighted = [UE3_weighted,UE4_weighted];
UE3_weighted_2_1 = UE3_UE4_weightTable2_1{:,1};
UE4_weighted_2_1 = UE3_UE4_weightTable2_1{:,2};
UE3_UE4_weighted_2_1 = [UE3_weighted_2_1,UE4_weighted_2_1];
UE3_weighted_2_2 = UE3_UE4_weightTable2_2{:,1};
UE4_weighted_2_2 = UE3_UE4_weightTable2_2{:,2};
UE3_UE4_weighted_2_2 = [UE3_weighted_2_2,UE4_weighted_2_2];

% Euclidean distance
UE3_UE4_Euclidean = pdist(UE3_UE4_weighted','euclidean');

% Cos = 1 - pdist(rAB','cosine')
UE3_UE4_Cos_w = 1 - pdist(UE3_UE4_weighted','cosine');
UE3_UE4_Cos_w2_1 = 1 - pdist(UE3_UE4_weighted_2_1','cosine');
UE3_UE4_Cos_w2_2 = 1 - pdist(UE3_UE4_weighted_2_2','cosine');

UE3_UE4_Similarity = (UE3_UE4_Cos_w2_1 + UE3_UE4_Cos_w2_2 ) / 2;

%%%%%%%%%%% similarity of UE1 & UE3 %%%%%%%%%%%
UE1_relative_frequency = UE1_UE3_TOTAL{:,2};
UE3_relative_frequency = UE1_UE3_TOTAL{:,4};
% rAB = [UE1_relative_frequency,UE2_relative_frequency];
UE1_weighted = UE1_UE3_TOTAL{:,5};
UE3_weighted = UE1_UE3_TOTAL{:,6};
UE1_UE3_weighted = [UE1_weighted,UE3_weighted];
UE1_weighted_2_1 = UE1_UE3_weightTable2_1{:,1};
UE3_weighted_2_1 = UE1_UE3_weightTable2_1{:,2};
UE1_UE3_weighted_2_1 = [UE1_weighted_2_1,UE3_weighted_2_1];
UE1_weighted_2_2 = UE1_UE3_weightTable2_2{:,1};
UE3_weighted_2_2 = UE1_UE3_weightTable2_2{:,2};
UE1_UE3_weighted_2_2 = [UE1_weighted_2_2,UE3_weighted_2_2];


% Euclidean distance
UE1_UE3_Euclidean = pdist(UE1_UE3_weighted','euclidean');

% Cos = 1 - pdist(rAB','cosine')
UE1_UE3_Cos_w = 1 - pdist(UE1_UE3_weighted','cosine');
UE1_UE3_Cos_w2_1 = 1 - pdist(UE1_UE3_weighted_2_1','cosine');
UE1_UE3_Cos_w2_2 = 1 - pdist(UE1_UE3_weighted_2_2','cosine');
UE1_UE3_Similarity = (UE1_UE3_Cos_w2_1 + UE1_UE3_Cos_w2_2 ) / 2;

%%%%%%%%%%% similarity of UE2 & UE3 %%%%%%%%%%%
UE2_relative_frequency = UE2_UE3_TOTAL{:,2};
UE3_relative_frequency = UE2_UE3_TOTAL{:,4};
% rAB = [UE1_relative_frequency,UE2_relative_frequency];
UE2_weighted = UE2_UE3_TOTAL{:,5};
UE3_weighted = UE2_UE3_TOTAL{:,6};
UE2_UE3_weighted = [UE2_weighted,UE3_weighted];
UE2_weighted_2_1 = UE2_UE3_weightTable2_1{:,1};
UE3_weighted_2_1 = UE2_UE3_weightTable2_1{:,2};
UE2_UE3_weighted_2_1 = [UE2_weighted_2_1,UE3_weighted_2_1];
UE2_weighted_2_2 = UE2_UE3_weightTable2_2{:,1};
UE3_weighted_2_2 = UE2_UE3_weightTable2_2{:,2};
UE2_UE3_weighted_2_2 = [UE2_weighted_2_2,UE3_weighted_2_2];


% Euclidean distance
UE2_UE3_Euclidean = pdist(UE2_UE3_weighted','euclidean');

% Cos = 1 - pdist(rAB','cosine')
UE2_UE3_Cos_w = 1 - pdist(UE2_UE3_weighted','cosine');
UE2_UE3_Cos_w2_1 = 1 - pdist(UE2_UE3_weighted_2_1','cosine');
UE2_UE3_Cos_w2_2 = 1 - pdist(UE2_UE3_weighted_2_2','cosine');
UE2_UE3_Similarity = (UE2_UE3_Cos_w2_1 + UE2_UE3_Cos_w2_2 ) / 2;

SimilarityResult =...
    [UE1_UE2_Similarity UE3_UE4_Similarity UE1_UE3_Similarity UE2_UE3_Similarity...
     Jaccard_index_UE1_UE2 Jaccard_index_UE3_UE4 Jaccard_index_UE1_UE3 Jaccard_index_UE2_UE3 ...
     UE1_UE2_Euclidean UE3_UE4_Euclidean UE1_UE3_Euclidean UE2_UE3_Euclidean];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 
% 
% figure(1)
% recordplot(cycle1_table)
fclose('all');


function RSSI_TABLE = rssiTable(data)
    for x = 1:10
        T{x} = table(data{1,x}(:,1),str2double(data{1,x}(:,2)), 'VariableNames',{'BSSID','RSSI'});
    end
    RSSI_TABLE = vertcat(T{:});
end

function newTable = modifyTable(data)
    [~,n] = size(data);
    for x = 1:n
        [a,~] = size(data{1,x}(:,2));
        ori = cell2mat(data{1,x}(:,2))
        cal_weight_RSSI = weightCal(ori');
        T{x} = table(data{1,x}(:,1),ori,ones(a,1),cal_weight_RSSI', 'VariableNames',{'BSSID','RSSI','count','weight'});
    end
    A = vertcat(T{:});

    [~,ind] = unique(A{:,1});
    for index = 1 : size(ind)
        sameind = find(ismember(A{:,1},A{ind(index),1}));
        sorted{index,1} = A{ind(index),1};
        sorted{index,2} = sum(A{sameind(:),3});
        sorted{index,3} = sum(A{sameind(:),4});
        sorted{index,4} = A{sameind(:),2};
    end
    newTable = cell2table(sorted);

end

function newTable = timeDividedTable(data,sub)
[~,n] = size(data);
    if sub == 1
     for x = 1:floor(n/2)
         [a,~] = size(data{1,x}(:,2));
         ori = cell2mat(data{1,x}(:,2));
         cal_weight_RSSI = weightCal(ori');
         T{x} = table(data{1,x}(:,1),ori,ones(a,1),cal_weight_RSSI', 'VariableNames',{'BSSID','RSSI','count','weight'});
     end
    elseif sub == 2
        for x = floor(n/2)+1 : n
            [a,~] = size(data{1,x}(:,2));
            ori = cell2mat(data{1,x}(:,2));
            cal_weight_RSSI = weightCal(ori');
            T{x} = table(data{1,x}(:,1),ori,ones(a,1),cal_weight_RSSI', 'VariableNames',{'BSSID','RSSI','count','weight'});
        end
    
    end
        A = vertcat(T{:});

      [~,ind] = unique(A{:,1});
     for index = 1 : size(ind)
          sameind = find(ismember(A{:,1},A{ind(index),1}));
        sorted{index,1} = A{ind(index),1};
        sorted{index,2} = sum(A{sameind(:),3});
        sorted{index,3} = sum(A{sameind(:),4});
        sorted{index,4} = A{sameind(:),2};
     end
    newTable = cell2table(sorted,'VariableNames',{'BSSID','count','weighted','RSSI'});

end

function weightByRSSI = weightCal(RSSI)
RSSI_MAX = -55.0;
RSSI_MIN = -100.0;
[~ , n] = size(RSSI);
    for x = 1:n
     if RSSI(x) > RSSI_MAX
           weightByRSSI(x) = 1;
       elseif RSSI(x) < RSSI_MIN
           weightByRSSI(x) = 0;
     else
           weightByRSSI(x) = (RSSI(x)-RSSI_MIN)/(RSSI_MAX-RSSI_MIN);
     end
    end

end

function Jaccard = Jaccard_cal(A1_AP,Red_AP)
union_AP = union (A1_AP, Red_AP);
intersect_AP = intersect(A1_AP, Red_AP);

Jaccard = numel(intersect_AP ) / numel(union_AP);
end

function [BSSID,count] = readRecords(file)
fid = fopen(file,'r');
data_scan = textscan(fid, '%s%f');
BSSID = data_scan{1,1};
count = data_scan{1,2};

end

function subCell = splitRecords(file)

fid = fopen(file,'r');
records = textscan(fid, '%s%s%s%s%s%s%s%s%s');
total_scan = sum(count(records{1,1},'Total:'));

A = horzcat(records{:});
index = find(contains(A(:,1),'Total:'));
size = str2double(A(index,2));

%%%%%% in case of 3 cycles => set m = 1:15 
%%%%%% in case of 2 cycles => set m = 1:10
%%%%%% otherwise m = 1:total_scan
for m=1:total_scan

idx = index(m)+1;
x=size(m);
% size_index = size(i)+1;
 subCell{m}=A(idx: idx+x-1,:);
 
end
fclose(fid);

end

function combineTable = combineWeightedTable(AP1,AP1_count,AP2,AP2_count)
totalAP = union(AP1,AP2);
total_size = numel(totalAP);
% index = [1:total_size]';
AP1_origin_size = numel(AP1);
AP2_origin_size = numel(AP2);

diff1 = setdiff(AP2,AP1); % AP2-AP1
diff2 = setdiff(AP1,AP2); % AP1-AP2
diff_AP1 = total_size - AP1_origin_size;
diff_AP2 = total_size - AP2_origin_size;

AP1_temp = cell(diff_AP1,1);
AP1_temp(:,1) = diff1;
AP1_temp_count = zeros(diff_AP1,1);
AP1_count = [AP1_count;AP1_temp_count];
AP1 = [AP1;AP1_temp];

AP2_temp = cell(diff_AP2,1);
AP2_temp(:,1) = diff2;
AP2_temp_count = zeros(diff_AP2,1);
AP2_count = [AP2_count;AP2_temp_count];
AP2 = [AP2 ; AP2_temp];


AP1_table = table(AP1_count, 'RowNames', AP1,'VariableNames',{'d1_weighted'});
AP2_table = table(AP2_count, 'RowNames', AP2,'VariableNames',{'d2_weighted'});
combineTable = join(AP1_table,AP2_table,'Keys','RowNames');

end

function combineTable = combine(AP1,AP1_count,AP2,AP2_count)
totalAP = union(AP1,AP2);
total_size = numel(totalAP);
% index = [1:total_size]';
AP1_origin_size = numel(AP1);
AP2_origin_size = numel(AP2);

diff1 = setdiff(AP2,AP1); % AP2-AP1
diff2 = setdiff(AP1,AP2); % AP1-AP2
diff_AP1 = total_size - AP1_origin_size;
diff_AP2 = total_size - AP2_origin_size;

AP1_temp = cell(diff_AP1,1);
AP1_temp(:,1) = diff1;
AP1_temp_count = zeros(diff_AP1,1);
AP1_count = [AP1_count;AP1_temp_count];
AP1 = [AP1;AP1_temp];

AP2_temp = cell(diff_AP2,1);
AP2_temp(:,1) = diff2;
AP2_temp_count = zeros(diff_AP2,1);
AP2_count = [AP2_count;AP2_temp_count];
AP2 = [AP2 ; AP2_temp];

total1 = sum(AP1_count);
total2 = sum(AP2_count);
% AP1_prob = AP1_count / AP1_total_scan ;
% AP2_prob = AP2_count / AP2_total_scan ; 
AP1_freq = AP1_count ;
AP2_freq = AP2_count ; 
AP1_relative_freq = AP1_count / total1 ;
AP2_relative_freq = AP2_count / total2 ; 
AP1_table = table(AP1_freq,AP1_relative_freq, 'RowNames', AP1,'VariableNames',{'d1_freq','d1_relative_feq'});
AP2_table = table(AP2_freq,AP2_relative_freq, 'RowNames', AP2,'VariableNames',{'d2_freq','d2_relative_feq'});
combineTable = join(AP1_table,AP2_table,'Keys','RowNames');

end

function nUE = noRepeatBSSID(UE)
[m n] = size(UE);

    for i = 1 : n
    nnUE = removeDupli(UE,i);
    nUE{1,i} = nnUE;

    end

end

function sorted = removeDupli(UE,n)

sortedUE = sortrows(UE{1,n});
 [~,ind] = unique(sortedUE(:,1));

    for index = 1 : size(ind)      
        sameind = find(ismember(sortedUE(:,1),sortedUE(ind(index))));
        NumOfsameind = numel(sameind);
        if NumOfsameind >=2
           meanRSSI = mean(str2double(sortedUE(sameind,2)));
           sorted{index,1} = sortedUE{ind(index),1};
           sorted{index,2} = meanRSSI;
           sorted{index,3} = sortedUE{ind(index),3};
           sorted{index,4} = sortedUE{ind(index),4};
        else
            sorted{index,1} = sortedUE{ind(index),1};
            sorted{index,2} = str2double(sortedUE(ind(index),2));
            sorted{index,3} = sortedUE{ind(index),3};
            sorted{index,4} = sortedUE{ind(index),4};
        end
    end
end

function recordplot(table1)

% A1_freq = table1{:,1}; % original frequency
% A1_freq2 = table1{:,2}; % relative frequency
% Red_freq = table1{:,3}; % original frequency
% Red_freq2 = table1{:,3};% relative frequency
% Red_non_c1 = 1 - Red_c1;
% Red_bar_c1 = [Red_c1';Red_non_c1'];
% Red_bar_c1 = [Red_c1'];
% total_c1 = height(table1);
% x = 1 : total_c1;
figure(1)
bar(table1{1:end,1:2:3})
legend('A1','RedMi');
xlabel('BSSID');
ylabel('Frequency');
set(gca,'fontsize',12)
title('Frequency Distribution')
print(gcf,'-r300','-dpng','freq_bar.png')

figure(2)
bar(table1{1:end,2:2:4})
legend('A1','RedMi');
xlabel('BSSID');
ylabel('Relative Frequency');
set(gca,'fontsize',12)
title('Relative Frequency Distribution')
print(gcf,'-r300','-dpng','relative_bar.png')

figure(3)
bar(table1{1:end,5:6})
legend('A1','RedMi');
xlabel('BSSID');
ylabel('Weighted-Frequency');
set(gca,'fontsize',12)
title('Weighted-Frequency Distribution')
print(gcf,'-r300','-dpng','weighted.png')


end
% 

