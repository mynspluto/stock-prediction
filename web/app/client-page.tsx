"use client";
import dynamic from "next/dynamic";

const DynamicChart = dynamic(() => import("../components/StockChart"), {
  ssr: false,
  loading: () => <div>Loading chart...</div>,
});

export default function ClientPage() {
  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <DynamicChart />
    </div>
  );
}
