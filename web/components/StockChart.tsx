"use client";
import React, { useEffect, useRef, useState } from "react";
import { createChart } from "lightweight-charts";
import axios from "axios";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";

// dayjs 플러그인 설정
dayjs.extend(utc);
dayjs.extend(timezone);

const CandlestickChart = () => {
  const chartRef = useRef(null);
  const [stockData, setStockData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [predictionData, setPredictionData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(
          "http://minikube.com:30000/predict/%5EIXIC"
        );

        // 날짜 형식 변환 함수
        const formatDate = (dateString) => {
          return dayjs(dateString).format("YYYY-MM-DD");
        };

        // historical_data를 차트 데이터 형식으로 변환
        const formattedData = response.data.historical_data.map((item) => ({
          time: formatDate(item.Date),
          open: item.Open,
          high: item.High,
          low: item.Low,
          close: item.Close,
          volume: item.Volume,
        }));

        // 데이터를 날짜순으로 정렬
        formattedData.sort((a, b) => dayjs(a.time).diff(dayjs(b.time)));

        setStockData(formattedData);
        setPredictionData({
          predictedClose: response.data.predicted_close,
          currentClose: response.data.current_close,
          predictionDate: formatDate(response.data.prediction_date),
        });
        setIsLoading(false);
      } catch (err) {
        console.error("Error fetching stock data:", err);
        setError(err.message);
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    if (!chartRef.current || isLoading || stockData.length === 0) return;

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 600,
      layout: {
        background: { color: "white" },
        textColor: "black",
      },
      rightPriceScale: {
        borderColor: "#D1D4DC",
        scaleMargins: {
          top: 0.1,
          bottom: 0.1, // 아래쪽 여백 증가
        },
      },
      timeScale: {
        borderColor: "#D1D4DC",
        timeVisible: true,
        rightOffset: 12,
        barSpacing: 60,
        fixLeftEdge: true,
        fixRightEdge: true,
        visible: true,
      },
      grid: {
        horzLines: {
          color: "#F0F3FA",
        },
        vertLines: {
          color: "#F0F3FA",
        },
      },
      crosshair: {
        mode: 1,
      },
    });

    // 캔들스틱 시리즈 추가
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
      priceFormat: {
        type: "price",
        precision: 0,
        minMove: 50, // 가격 표시 간격 조정
      },
    });

    candleSeries.priceScale().applyOptions({
      // set the positioning of the volume series
      scaleMargins: {
        top: 0.1, // highest point of the series will be 70% away from the top
        bottom: 0.1,
      },
    });

    // 데이터 설정
    candleSeries.setData(stockData);

    // 거래량 시리즈 추가 부분 수정
    const volumeSeries = chart.addHistogramSeries({
      color: "#26a69a",
      priceFormat: {
        type: "volume",
      },
      priceScaleId: "volume", // 별도의 price scale ID 설정,
      priceLineVisible: false,
      baseLineVisible: false,
      lastValueVisible: false,
    });

    // 거래량 데이터 설정
    const volumeData = stockData.map((item) => ({
      time: item.time,
      value: item.volume,
      color: item.close >= item.open ? "#26a69a55" : "#ef535055",
    }));
    volumeSeries.setData(volumeData);

    // 거래량 차트의 price scale 설정 수정
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.9,
        bottom: 0,
      },
    });

    // 예측값 마커 추가 (있는 경우)
    if (predictionData) {
      candleSeries.createPriceLine({
        price: predictionData.predictedClose,
        color: "#2962FF",
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: true,
        title: "Predicted",
      });
    }

    // 차트 컨텐츠에 맞게 자동 조정
    chart.timeScale().fitContent();

    // 차트 크기 조절
    const handleResize = () => {
      chart.applyOptions({
        width: chartRef.current.clientWidth,
      });
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [isLoading, stockData, predictionData]);

  if (isLoading) {
    return <div className="text-center p-4">Loading...</div>;
  }

  if (error) {
    return <div className="text-red-500 p-4">Error: {error}</div>;
  }

  const calculateChange = () => {
    if (!predictionData) return null;
    const change = predictionData.predictedClose - predictionData.currentClose;
    const percentChange = (change / predictionData.currentClose) * 100;
    return {
      change,
      percentChange,
      isPositive: change > 0,
    };
  };

  const changeInfo = calculateChange();

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">NASDAQ Index</h2>
        {predictionData && (
          <div className="mb-4 text-sm grid grid-cols-2 gap-4">
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-gray-600">Current Close</div>
              <div className="text-lg font-semibold">
                $
                {predictionData.currentClose.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </div>
            </div>
            <div
              className={`p-3 rounded-lg ${
                changeInfo?.isPositive ? "bg-green-50" : "bg-red-50"
              }`}
            >
              <div
                className={`${
                  changeInfo?.isPositive ? "text-green-600" : "text-red-600"
                }`}
              >
                Predicted Close
              </div>
              <div
                className={`text-lg font-semibold ${
                  changeInfo?.isPositive ? "text-green-700" : "text-red-700"
                }`}
              >
                $
                {predictionData.predictedClose.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
                <span className="text-sm ml-2">
                  ({changeInfo?.isPositive ? "+" : ""}
                  {changeInfo?.percentChange.toFixed(2)}%)
                </span>
              </div>
            </div>
            <div className="col-span-2 text-center text-gray-600">
              다음 영업일 나스닥 종합지수 예측
            </div>
          </div>
        )}
        <div ref={chartRef} className="w-full" />
      </div>
    </div>
  );
};

export default CandlestickChart;
