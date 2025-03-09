"use client";
import dynamic from "next/dynamic";

// ApiResponse 타입 정의
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

// 컴포넌트 props 타입 정의
interface ClientPageProps {
  initialData: ApiResponse | null;
}

// dynamic import는 클라이언트 컴포넌트에서만 ssr: false 옵션 사용 가능
const DynamicChart = dynamic(() => import("../components/StockChart"), {
  ssr: false,
  loading: () => <div>Loading chart...</div>,
});

export default function ClientPage({ initialData }: ClientPageProps) {
  return (
    <div className="min-h-screen bg-gray-100 py-8">
      {initialData ? (
        <DynamicChart initialData={initialData} />
      ) : (
        <DynamicChart />
      )}
    </div>
  );
}
