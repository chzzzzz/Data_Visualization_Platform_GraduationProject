import * as echarts from 'echarts';

/*
var ROOT_PATH = 'https://echarts.apache.org/examples';
*/
var chartDom = document.getElementById('main');
var myChart = echarts.init(chartDom);
var option;

$.get(
   'http://127.0.0.1:5000/fileUpLoad',
  function (_rawData) {
    run(_rawData);
  }
);
function run(_rawData) {
  // var countries = ['Australia', 'Canada', 'China', 'Cuba', 'Finland', 'France', 'Germany', 'Iceland', 'India', 'Japan', 'North Korea', 'South Korea', 'New Zealand', 'Norway', 'Poland', 'Russia', 'Turkey', 'United Kingdom', 'United States'];
  print('run');
  const countries = [
    '负载电流',
    '负载电压',
    '逆变电流',
    '逆变电压',
    '输入电流',
    '输入电压'
  ];
  const dic = {
    '负载电流':'fz_i',
    '负载电压':'fz_v',
    '逆变电流':'nb_i',
    '逆变电压':'nb_v',
    '输入电流':'sr_i',
    '输入电压':'sr_v'
  }
  const datasetWithFilters = [];
  const seriesList = [];
  echarts.util.each(countries, function (country) {
    var datasetId = 'dataset_' + dic[country];
    datasetWithFilters.push({
      id: datasetId,
      fromDatasetId: 'dataset_raw',
      transform: {
        type: 'filter',
        config: {
          and: [
            { dimension: 'type', '=': country }
          ]
        }
      }
    });
    seriesList.push({
      type: 'line',
      datasetId: datasetId,
      showSymbol: false,
      name: country,
      endLabel: {
        show: true,
        formatter: function (params) {
          return params.value[1] + ': ' + params.value[2];
        }
      },
      labelLayout: {
        moveOverlap: 'shiftY'
      },
      emphasis: {
        focus: 'series'
      },
      encode: {
        x: 'time',
        y: 'value',
        label: ['type', 'value'],
        itemName: 'time',
        tooltip: ['type']
      }
    });
  });
  option = {
    animationDuration: 10000,
    dataset: [
      {
        id: 'dataset_raw',
        source: _rawData
      },
      ...datasetWithFilters
    ],
    title: {
      text: '数据初步展示'
    },
    tooltip: {
      order: 'valueDesc',
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      nameLocation: 'middle'
    },
    yAxis: {
      name: 'value'
    },
    grid: {
      right: 140
    },
    series: seriesList
  };
  myChart.setOption(option);
}
option && myChart.setOption(option);

