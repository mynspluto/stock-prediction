"use client";
import React, { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";

const CandlestickChart = () => {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!chartRef.current) return;

    const chart = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: "white" },
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
        mode: 1,
      },
    });

    // 데이터 준비
    const candleData = [
      {
        open: 150.25,
        high: 153.5,
        low: 149.75,
        close: 152.5,
        time: "2024-01-01",
      },
      {
        open: 152.5,
        high: 155.25,
        low: 151.0,
        close: 153.75,
        time: "2024-01-02",
      },
      {
        open: 153.75,
        high: 154.5,
        low: 148.25,
        close: 149.75,
        time: "2024-01-03",
      },
      {
        open: 149.75,
        high: 156.0,
        low: 149.0,
        close: 155.25,
        time: "2024-01-04",
      },
      {
        open: 155.25,
        high: 158.75,
        low: 154.5,
        close: 157.0,
        time: "2024-01-05",
      },
    ];

    try {
      // 캔들스틱 시리즈 추가
      const candleSeries = chart.addCandlestickSeries({
        upColor: "#26a69a",
        downColor: "#ef5350",
        borderVisible: false,
        wickUpColor: "#26a69a",
        wickDownColor: "#ef5350",
      });

      // 데이터 설정
      candleSeries.setData(candleData);

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
    } catch (error) {
      console.error("Error creating chart:", error);
    }
  }, []);

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Stock Price Chart</h2>
        <div ref={chartRef} className="w-full" />
      </div>
    </div>
  );
};

export default CandlestickChart;
