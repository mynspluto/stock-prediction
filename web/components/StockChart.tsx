"use client";
import React, { useEffect, useRef, useState } from "react";
import { createChart, ColorType, CrosshairMode } from "lightweight-charts";
import axios from "axios";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";
import dotenv from "dotenv";

dotenv.config();

dayjs.extend(utc);
dayjs.extend(timezone);

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

console.log("process.env", process.env);
console.log("API_BASE_URL", API_BASE_URL);
interface FormattedData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ApiResponse {
  historical_data: {
    Date: string;
    Open: string;
    High: string;
    Low: string;
    Close: string;
    Volume: string;
  }[];
  predicted_close: number;
  current_close: number;
  prediction_date: string;
}

interface PredictionData {
  predictedClose: number;
  currentClose: number;
  predictionDate: string;
}

interface ChangeInfo {
  change: number;
  percentChange: number;
  isPositive: boolean;
}

interface StockChartProps {
  initialData?: ApiResponse | null;
}

const CandlestickChart = ({ initialData }: StockChartProps) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const [stockData, setStockData] = useState<FormattedData[]>([]);
  const [isLoading, setIsLoading] = useState(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(
    null
  );

  useEffect(() => {
    // initialData가 있으면 바로 사용
    if (initialData) {
      const formatDate = (dateString: string): string => {
        return dayjs(dateString).format("YYYY-MM-DD");
      };

      const formattedData: FormattedData[] = initialData.historical_data.map(
        (item) => ({
          time: formatDate(item.Date),
          open: parseFloat(item.Open),
          high: parseFloat(item.High),
          low: parseFloat(item.Low),
          close: parseFloat(item.Close),
          volume: parseFloat(item.Volume),
        })
      );

      formattedData.sort((a, b) => dayjs(a.time).diff(dayjs(b.time)));

      setStockData(formattedData);
      setPredictionData({
        predictedClose: initialData.predicted_close,
        currentClose: initialData.current_close,
        predictionDate: formatDate(initialData.prediction_date),
      });
      setIsLoading(false);
      return;
    }

    // initialData가 없으면 기존 방식으로 데이터 가져오기
    const fetchData = async () => {
      try {
        const response = await axios.get<ApiResponse>(
          `${API_BASE_URL}/predict/%5EIXIC`
        );

        const formatDate = (dateString: string): string => {
          return dayjs(dateString).format("YYYY-MM-DD");
        };

        const formattedData: FormattedData[] =
          response.data.historical_data.map((item) => ({
            time: formatDate(item.Date),
            open: parseFloat(item.Open),
            high: parseFloat(item.High),
            low: parseFloat(item.Low),
            close: parseFloat(item.Close),
            volume: parseFloat(item.Volume),
          }));

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
        setError(err instanceof Error ? err.message : "An error occurred");
        setIsLoading(false);
      }
    };

    fetchData();
  }, [initialData]);

  useEffect(() => {
    if (!chartRef.current || isLoading || stockData.length === 0) return;

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 600,
      layout: {
        background: { type: ColorType.Solid, color: "white" },
        textColor: "black",
      },
      rightPriceScale: {
        borderColor: "#D1D4DC",
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
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
        mode: CrosshairMode.Normal,
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
      priceFormat: {
        type: "price",
        precision: 0,
        minMove: 50,
      },
    });

    candleSeries.setData(stockData);

    const volumeSeries = chart.addHistogramSeries({
      color: "#26a69a",
      priceFormat: {
        type: "volume",
      },
      priceScaleId: "volume",
      priceLineVisible: false,
      baseLineVisible: false,
      lastValueVisible: false,
    });

    const volumeData = stockData.map((item) => ({
      time: item.time,
      value: item.volume,
      color: item.close >= item.open ? "#26a69a55" : "#ef535055",
    }));

    volumeSeries.setData(volumeData);

    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.9,
        bottom: 0,
      },
    });

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

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartRef.current) {
        chart.applyOptions({
          width: chartRef.current.clientWidth,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [isLoading, stockData, predictionData]);

  const calculateChange = (): ChangeInfo | null => {
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

  if (isLoading) {
    return <div className="text-center p-4">Loading...</div>;
  }

  if (error) {
    return <div className="text-red-500 p-4">Error: {error}</div>;
  }

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">나스닥 지수</h2>
        {predictionData && (
          <div className="mb-4 text-sm grid grid-cols-2 gap-4">
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-gray-600">최근 종가</div>
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
