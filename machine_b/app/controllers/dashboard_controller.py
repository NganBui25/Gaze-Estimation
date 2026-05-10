from datetime import date
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette import status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.dashboard_service import DashboardService


router = APIRouter(tags=["dashboard"])

templates = Jinja2Templates(directory="app/templates")


@router.get("/dashboard", response_class=HTMLResponse)
def overview(
    request: Request,
    db: Session = Depends(get_db),
):
    dashboard_service = DashboardService(db)
    overview_data = dashboard_service.get_overview(trend_days=7)

    return templates.TemplateResponse(
        request=request,
        name="dashboard/overview.html",
        context={
            "page_title": "Dashboard Overview",
            "page_subtitle": "Advertisement monitoring and performance summary",
            "active_page": "dashboard",
            "overview": overview_data,
        },
    )


@router.get("/dashboard/advertisements", response_class=HTMLResponse)
def advertisements_page(
    request: Request,
    search: str | None = Query(default=None),
    category_id: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    message: str | None = Query(default=None),
    message_type: str | None = Query(default="success"),
    db: Session = Depends(get_db),
):
    normalized_category_id = _parse_optional_int(category_id)

    dashboard_service = DashboardService(db)
    advertisements = dashboard_service.get_advertisements_page_data(
        search_term=search,
        category_id=normalized_category_id,
        status_filter=status_filter,
    )
    categories = dashboard_service.get_category_options()

    return templates.TemplateResponse(
        request=request,
        name="dashboard/advertisements.html",
        context={
            "page_title": "Advertisements",
            "page_subtitle": "Manage advertisement inventory and activation status",
            "active_page": "advertisements",
            "advertisements": advertisements,
            "categories": categories,
            "filters": {
                "search": search or "",
                "category_id": normalized_category_id,
                "status": status_filter or "",
            },
            "message": message,
            "message_type": message_type or "success",
        },
    )


@router.post("/dashboard/advertisements/create")
def create_advertisement(
    title: str = Form(...),
    description: str | None = Form(default=None),
    media_filename: str = Form(...),
    duration_seconds: int = Form(...),
    category_id: int = Form(...),
    db: Session = Depends(get_db),
):
    dashboard_service = DashboardService(db)

    try:
        dashboard_service.create_advertisement(
            title=title,
            description=description,
            media_filename=media_filename,
            duration_seconds=duration_seconds,
            category_id=category_id,
        )
        return _redirect_to_advertisements(
            message="Advertisement created successfully",
            message_type="success",
        )
    except ValueError as exc:
        db.rollback()
        return _redirect_to_advertisements(
            message=str(exc),
            message_type="danger",
        )


@router.post("/dashboard/advertisements/{advertisement_id}/edit")
def edit_advertisement(
    advertisement_id: int,
    title: str = Form(...),
    description: str | None = Form(default=None),
    media_filename: str = Form(...),
    duration_seconds: int = Form(...),
    category_id: int = Form(...),
    db: Session = Depends(get_db),
):
    dashboard_service = DashboardService(db)

    try:
        dashboard_service.update_advertisement(
            advertisement_id=advertisement_id,
            title=title,
            description=description,
            media_filename=media_filename,
            duration_seconds=duration_seconds,
            category_id=category_id,
        )
        return _redirect_to_advertisements(
            message="Advertisement updated successfully",
            message_type="success",
        )
    except (LookupError, ValueError) as exc:
        db.rollback()
        return _redirect_to_advertisements(
            message=str(exc),
            message_type="danger",
        )


@router.post("/dashboard/advertisements/{advertisement_id}/toggle")
def toggle_advertisement(
    advertisement_id: int,
    db: Session = Depends(get_db),
):
    dashboard_service = DashboardService(db)

    try:
        is_active = dashboard_service.toggle_advertisement(advertisement_id)
        message = "Advertisement activated" if is_active else "Advertisement deactivated"
        return _redirect_to_advertisements(
            message=message,
            message_type="success",
        )
    except LookupError as exc:
        db.rollback()
        return _redirect_to_advertisements(
            message=str(exc),
            message_type="danger",
        )


@router.post("/dashboard/advertisements/{advertisement_id}/delete")
def delete_advertisement(
    advertisement_id: int,
    db: Session = Depends(get_db),
):
    dashboard_service = DashboardService(db)

    try:
        dashboard_service.delete_advertisement(advertisement_id)
        return _redirect_to_advertisements(
            message="Advertisement deleted successfully",
            message_type="success",
        )
    except (LookupError, ValueError) as exc:
        db.rollback()
        return _redirect_to_advertisements(
            message=str(exc),
            message_type="danger",
        )


@router.get("/dashboard/categories", response_class=HTMLResponse)
def categories_page(
    request: Request,
    db: Session = Depends(get_db),
):
    dashboard_service = DashboardService(db)
    categories = dashboard_service.get_categories_page_data()

    return templates.TemplateResponse(
        request=request,
        name="dashboard/categories.html",
        context={
            "page_title": "Categories",
            "page_subtitle": "Review advertisement groups and their audience fit",
            "active_page": "categories",
            "categories": categories,
        },
    )


@router.get("/dashboard/play-logs", response_class=HTMLResponse)
def play_logs_page(
    request: Request,
    db: Session = Depends(get_db),
):
    dashboard_service = DashboardService(db)
    play_logs = dashboard_service.get_play_logs_page_data(limit=50)
    advertisements = dashboard_service.get_advertisement_options()

    return templates.TemplateResponse(
        request=request,
        name="dashboard/play_logs.html",
        context={
            "page_title": "Play Logs",
            "page_subtitle": "Browse advertisement playback history",
            "active_page": "play_logs",
            "play_logs": play_logs,
            "advertisements": advertisements,
        },
    )


@router.get("/dashboard/reports", response_class=HTMLResponse)
def reports_page(
    request: Request,
    period: str = Query(default="daily"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    db: Session = Depends(get_db),
):
    dashboard_service = DashboardService(db)
    report_summary = dashboard_service.get_filtered_report_summary(
        period=period,
        date_from=date_from,
        date_to=date_to,
    )
    report_rows = dashboard_service.get_report_rows(
        limit=100,
        period=period,
        date_from=date_from,
        date_to=date_to,
    )

    return templates.TemplateResponse(
        request=request,
        name="dashboard/reports.html",
        context={
            "page_title": "Reports",
            "page_subtitle": "Summarize advertisement performance by period",
            "active_page": "reports",
            "report_summary": report_summary,
            "report_rows": report_rows,
            "filters": {
                "period": period,
                "date_from": date_from.isoformat() if date_from else "",
                "date_to": date_to.isoformat() if date_to else "",
            },
        },
    )


def _redirect_to_advertisements(
    message: str,
    message_type: str,
) -> RedirectResponse:
    query = urlencode(
        {
            "message": message,
            "message_type": message_type,
        }
    )
    return RedirectResponse(
        url=f"/dashboard/advertisements?{query}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    return int(normalized)
